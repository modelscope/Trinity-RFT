# -*- coding: utf-8 -*-
"""vLLM related modules.

Modified from OpenRLHF openrlhf/trainer/ray/vllm_engine.py
"""
from __future__ import annotations

import os
import re
import threading
from typing import List, Optional, Tuple

import torch
import vllm
from vllm import LLM
from vllm.sampling_params import SamplingParams

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.models.utils import (
    tokenize_and_mask_messages_default,
    tokenize_and_mask_messages_hf,
)
from trinity.utils.log import get_logger


class vLLMRolloutModel(InferenceModel):
    """Actor for vLLM."""

    def __init__(self, config: InferenceModelConfig):
        self.logger = get_logger(__name__)
        self.config = config
        if config.tensor_parallel_size != 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = config.bundle_indices
        if not vllm.envs.is_set("VLLM_USE_V1"):
            self.logger.info(f"Using vLLM v{int(config.use_v1)} engine")
            os.environ["VLLM_USE_V1"] = str(int(config.use_v1))
        if config.use_v1:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(int(config.use_v1))
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        self.default_sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=config.max_response_tokens,
            min_tokens=1,
            truncate_prompt_tokens=config.max_prompt_tokens,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            logprobs=0,
        )
        max_model_len = None
        if config.max_prompt_tokens is not None and config.max_response_tokens is not None:
            max_model_len = config.max_prompt_tokens + config.max_response_tokens
        self.llm = LLM(
            # TODO: check checkpoint path
            model=config.model_path,
            enforce_eager=config.enforce_eager,
            worker_extension_cls="trinity.common.models.vllm_worker.WorkerExtension",
            tensor_parallel_size=config.tensor_parallel_size,
            seed=config.seed,
            distributed_executor_backend=("uni" if config.tensor_parallel_size == 1 else "ray"),
            max_model_len=max_model_len,
            enable_prefix_caching=config.enable_prefix_caching,
            dtype=config.dtype,
            trust_remote_code=True,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_chunked_prefill=config.enable_chunked_prefill,
            # max_num_batched_tokens=256,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.chat_template = self.tokenizer.get_chat_template()
        self.enable_thinking = config.enable_thinking
        if self.config.chat_template:
            self.chat_template = self.config.chat_template
        if not re.search(r"\{\%-?\s*generation\s*-?\%\}", self.chat_template):
            self.logger.warning(
                "The provided chat template does not support `return_assitant_tokens_mask`. "
                "The default assistant mask method will be used, which may cause performance "
                "degradation and lead to incorrect results."
            )
            self.action_mask_method = tokenize_and_mask_messages_default
        else:
            self.action_mask_method = tokenize_and_mask_messages_hf
        self.lock = threading.Lock()
        self.state_dict_meta = None
        self.ckp_version = 0  # TODO: resume the value from the checkpoint

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        update_with_checkpoint: bool = True,
        state_dict_meta: dict = None,
    ):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
                timeout,
                update_with_checkpoint,
                state_dict_meta,
                self.config.ray_namespace,
            ),
        )

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def _create_sampling_params(self, **kwargs):
        """Create sampling params."""
        if len(kwargs) == 0:
            return self.default_sampling_params
        params = self.default_sampling_params.clone()
        for k, v in kwargs.items():
            if hasattr(params, k):
                setattr(params, k, v)
        return params

    def generate(self, prompts: List[str], **kwargs) -> List:
        """
        Generate a batch of responses from a batch of prompts.

        Note:

            This method will not apply chat template.
            You need to apply chat template before calling this method.

        Args:
            prompts (List[str]): A list of prompts.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            List[Experience]: A list of experiences.

        Example:

            >>> # config.algorithm.repeat_times == 2 or kwargs["n"] == 2
            >>>
            >>> prompts = [
            >>>     "Hello, world!",
            >>>     "How are you?"
            >>> ]
            >>> experiences = model.generate(prompts)
            >>> print(experiences)
            [
                Experience(tokens=tensor()...),  # first sequnece for prompts[0]
                Experience(tokens=tensor()...),  # second sequnece for prompts[0]
                Experience(tokens=tensor()...),  # first sequence for prompts[1]
                Experience(tokens=tensor()...)   # second sequence for prompts[1]
            ]
        """
        with self.lock:
            sampling_params = self._create_sampling_params(**kwargs)
            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        experiences = []
        for output in outputs:
            for i in range(sampling_params.n):
                experiences.append(
                    Experience(
                        tokens=torch.cat(
                            (
                                torch.tensor(output.prompt_token_ids, dtype=torch.int32),
                                torch.tensor(output.outputs[i].token_ids, dtype=torch.int32),
                            )
                        ),
                        logprobs=torch.cat(
                            (
                                torch.full(
                                    (len(output.prompt_token_ids),),
                                    0.0,
                                    dtype=torch.float32,
                                ),
                                torch.tensor(
                                    [
                                        list(logprob_dict.values())[0].logprob
                                        for logprob_dict in output.outputs[i].logprobs
                                    ],
                                    dtype=torch.float32,
                                ),
                            )
                        ),
                        prompt_length=len(output.prompt_token_ids),
                        prompt_text=output.prompt,
                        response_text=output.outputs[i].text,
                    )
                )
        return experiences

    def chat(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Chat with the model with a list of messages.

        Args:
            messages (List[dict]): A list of messages.

        Example:

            >>> [
            >>>   {"role": "system", "content": "You are a helpful assistant."},
            >>>   {"role": "user", "content": "Hello, world!"},
            >>> ]

        Returns:
            List[Experience]: A list of experiences containing the response text.
        """
        # TODO: support tools and other fields
        if messages[-1]["role"] == "assistant":
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
                chat_template=self.chat_template,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template,
                enable_thinking=self.enable_thinking,
            )
        return self.generate([prompt], **kwargs)

    def logprobs(self, token_ids: List[int]) -> torch.Tensor:
        with self.lock:
            outputs = self.llm.generate(
                prompts={"prompt_token_ids": token_ids},
                sampling_params=self._create_sampling_params(
                    n=1,
                    max_tokens=1,
                    prompt_logprobs=0,
                ),
                use_tqdm=False,
            )
        return torch.tensor(
            [0]
            + [
                list(logprob_dict.values())[0].logprob
                for logprob_dict in outputs[0].prompt_logprobs[1:]
            ],
            dtype=torch.float32,
        )

    def convert_messages_to_experience(self, messages: List[dict]) -> Experience:
        """Convert a list of messages into an experience."""
        token_ids, action_mask = self.action_mask_method(
            self.tokenizer, messages, self.chat_template
        )
        logprobs = self.logprobs(token_ids=token_ids.tolist())
        return Experience(
            tokens=token_ids,
            prompt_length=len(token_ids),
            logprobs=logprobs,
            action_mask=action_mask,
        )

    def has_api_server(self) -> bool:
        return False

    def sync_model(self, update_weight_args_list: Optional[List[Tuple]] = None) -> bool:
        """Sync model weights to vLLM."""
        if update_weight_args_list is not None:
            self._collective_rpc("set_state_dict_meta", args=(update_weight_args_list,))
        self._collective_rpc("update_weight")
        self.logger.info("Sync model weights to vLLM successfully.")
        self.ckp_version += 1
        return True

    def get_ckp_version(self) -> int:
        return self.ckp_version
