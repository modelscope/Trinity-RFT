# -*- coding: utf-8 -*-
"""vLLM related modules.

Modified from OpenRLHF openrlhf/trainer/ray/vllm_engine.py
"""
from __future__ import annotations

import os
import re
import threading
from typing import List

import ray
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.models.utils import (
    tokenize_and_mask_messages_default,
    tokenize_and_mask_messages_hf,
)
from trinity.utils.log import get_logger


@ray.remote
class vLLMRolloutModel(InferenceModel):
    """Actor for vLLM."""

    def __init__(self, config: Config, **kwargs):
        self.logger = get_logger(__name__)
        self.config = config
        if config.explorer.tensor_parallel_size != 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.default_sampling_params = SamplingParams(
            n=config.explorer.repeat_times,
            temperature=config.explorer.temperature,
            top_p=config.explorer.top_p,
            top_k=config.explorer.top_k,
            max_tokens=config.model.max_response_tokens,
            min_tokens=1,
            truncate_prompt_tokens=config.model.max_prompt_tokens,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            logprobs=config.explorer.logprobs,
        )
        self.llm = LLM(
            # TODO: check checkpoint path
            model=config.model.model_path,
            enforce_eager=config.explorer.enforce_eager,
            worker_cls="trinity.common.models.vllm_worker.VLLMWorker",
            tensor_parallel_size=config.explorer.tensor_parallel_size,
            seed=config.explorer.seed,
            distributed_executor_backend=(
                "uni" if config.explorer.tensor_parallel_size == 1 else "mp"
            ),
            max_model_len=config.model.max_prompt_tokens + config.model.max_response_tokens,
            enable_prefix_caching=config.explorer.enable_prefix_caching,
            dtype=config.explorer.dtype,
            trust_remote_code=True,
            gpu_memory_utilization=config.explorer.gpu_memory_utilization,
            enable_chunked_prefill=config.explorer.enable_chunked_prefil,
            # max_num_batched_tokens=256,
            **kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.chat_template = self.tokenizer.get_chat_template()
        if self.config.explorer.chat_template:
            self.chat_template = self.config.explorer.chat_template
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
            ),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
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

            >>> # config.explorer.repeat_times == 2 or kwargs["repeat_times"] == 2
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
            outputs = self.llm.generate(
                prompts, self._create_sampling_params(**kwargs), use_tqdm=False
            )
        experiences = []
        for output in outputs:
            for i in range(self.config.explorer.repeat_times):
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
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.chat_template,
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
        logprobs = self.logprobs(token_ids=token_ids)
        return Experience(
            tokens=token_ids,
            prompt_length=len(token_ids),
            logprobs=logprobs,
            action_mask=action_mask,
        )

    def sync_model(self, update_weight_args_list) -> bool:
        """Sync model weights to vLLM."""
        with self.lock:
            for args in update_weight_args_list:
                self.llm.collective_rpc("update_weight", args=args)
        self.logger.info("Sync model weights to vLLM successfully.")
        self.ckp_version += 1
        return True

    def get_ckp_version(self) -> int:
        return self.ckp_version
