"""vLLM AsyncEngine wrapper.

Modified from Ray python/ray/llm/_internal/batch/stages/vllm_engine_stage.py
"""

import asyncio
import os
import re
from contextlib import nullcontext
from typing import Any, Dict, List

import ray
import torch
import vllm

from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.models.utils import (
    tokenize_and_mask_messages_default,
    tokenize_and_mask_messages_hf,
)
from trinity.utils.log import get_logger

logger = get_logger(__name__)


# TODO: merge into vLLMRolloutModel
@ray.remote
class vLLMAysncRolloutModel(InferenceModel):
    """Wrapper around the vLLM engine to handle async requests.

    Args:
        config (Config): The config.
        kwargs (dict): The keyword arguments for the engine.
    """

    def __init__(
        self,
        config: Config,
        **kwargs,
    ) -> None:
        self.logger = get_logger(__name__)
        self.config = config
        if config.explorer.tensor_parallel_size != 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.default_sampling_params = vllm.SamplingParams(
            n=config.explorer.repeat_times,
            temperature=config.explorer.temperature,
            max_tokens=config.model.max_response_tokens,
            min_tokens=1,
            truncate_prompt_tokens=config.model.max_prompt_tokens,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            logprobs=config.explorer.logprobs,
        )
        self.request_id = 0
        engine_args = vllm.AsyncEngineArgs(
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
            task="generate",
            disable_log_requests=True,
            gpu_memory_utilization=config.explorer.gpu_memory_utilization,
            enable_chunked_prefill=config.explorer.enable_chunked_prefil,
            # max_num_batched_tokens=256, # you can further set this parameter to reduce the vllm peak memory usage
        )
        self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = None
        self.chat_template = None
        if self.config.explorer.chat_template:
            self.chat_template = self.config.explorer.chat_template
        if self.chat_template is None or not re.search(
            r"\{\%-?\s*generation\s*-?\%\}", self.chat_template
        ):
            self.logger.warning(
                "The provided chat template does not support `return_assitant_tokens_mask`. "
                "The default assistant mask method will be used, which may cause performance "
                "degradation and lead to incorrect results."
            )
            self.action_mask_method = tokenize_and_mask_messages_default
        else:
            self.action_mask_method = tokenize_and_mask_messages_hf
        # The performance gets really bad if there are too many requests in the pending queue.
        # We work around it with semaphore to limit the number of concurrent requests in the engine.
        self.max_pending_requests = config.explorer.max_pending_requests
        if self.max_pending_requests > 0:
            self.semaphore = asyncio.Semaphore(self.max_pending_requests)
        else:
            self.semaphore = nullcontext()
        self.ckp_version = 0  # TODO: resume the value from the checkpoint

    async def chat_async(self, messages: List[Dict], **kwargs) -> List[Experience]:
        """Chat with the model with a list of messages in async.

        Args:
            messages (List[dict]): The input history messages.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        if self.tokenizer is None:
            self.tokenizer = await self.async_llm.get_tokenizer()
        if self.chat_template is None:
            self.chat_template = self.tokenizer.get_chat_template()
        prompt = self.tokenizer.apply_chat_template(
            messages,
            chat_template=self.chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        return await self.generate_async(prompt=prompt, **kwargs)

    async def generate_async(self, prompt: str, **kwargs) -> List[Experience]:
        """Generate a response from the provided prompt in async.

        Args:
            prompt (str): The input prompt.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        request_id = self.request_id
        self.request_id += 1
        async with self.semaphore:
            output = await self._generate_internal(request_id=request_id, prompt=prompt, **kwargs)
        experiences = [
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
            for i in range(len(output.outputs))
        ]
        return experiences

    async def logprobs_async(self, token_ids: List[int]) -> torch.Tensor:
        """Calculate the logprobs of the given tokens in async."""
        request_id = self.request_id
        self.request_id += 1
        async with self.semaphore:
            output = await self._generate_internal(
                request_id=request_id,
                prompt={"prompt_token_ids": token_ids},
                n=1,
                max_tokens=1,
                prompt_logprobs=0,  # vLLM return `prompt_logprobs + 1` logrpobs for each token
            )
        return torch.tensor(
            [0]
            + [
                list(logprob_dict.values())[0].logprob
                for logprob_dict in output.prompt_logprobs[1:]
            ],
            dtype=torch.float32,
        )

    async def _generate_internal(self, request_id: int, prompt: Any, **kwargs) -> Any:
        # Send the request to the LLM engine.
        stream = self.async_llm.generate(
            request_id=str(request_id),
            prompt=prompt,
            sampling_params=self._create_sampling_params(**kwargs),
        )

        # Consume the stream until the request is finished.
        async for request_output in stream:
            if request_output.finished:
                # Bypass the original full prompt.
                # request_output.prompt = request.prompt
                return request_output

        raise RuntimeError(
            "[vLLM] The request is not finished. This should not happen. "
            "Please report this issue to the Ray team."
        )

    async def convert_messages_to_experience_async(self, messages: List[dict]) -> Experience:
        """Convert a list of messages into an experience."""
        if self.tokenizer is None:
            self.tokenizer = await self.async_llm.get_tokenizer()
        if self.chat_template is None:
            self.chat_template = self.tokenizer.get_chat_template()
        token_ids, action_mask = self.action_mask_method(
            self.tokenizer, messages, self.chat_template
        )
        logprobs = await self.logprobs_async(token_ids=token_ids)
        return Experience(
            tokens=token_ids,
            prompt_length=len(token_ids),
            logprobs=logprobs,
            action_mask=action_mask,
        )

    def shutdown(self):
        """Shutdown the vLLM v1 engine. This kills child processes forked
        by the vLLM engine. If not called, the child processes will be
        orphaned and will not be killed when the parent process exits,
        and they won't be able to be tracked by Ray anymore.
        """
        if hasattr(self.async_llm, "shutdown"):
            logger.info("Shutting down vLLM engine")
            self.async_llm.shutdown()

    def _create_sampling_params(self, **kwargs):
        """Create sampling params."""
        if len(kwargs) == 0:
            return self.default_sampling_params
        params = self.default_sampling_params.clone()
        for k, v in kwargs.items():
            if hasattr(params, k):
                setattr(params, k, v)
        return params

    def sync_model(self, update_weight_args_list) -> bool:
        """Sync model weights to vLLM."""
        for args in update_weight_args_list:
            self.async_llm.engine.model_executor.collective_rpc("update_weight", args=args)
        self.logger.info("Sync model weights to vLLM successfully.")
        self.ckp_version += 1
        return True

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
        return self.async_llm.engine.model_executor.collective_rpc(
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
        return self.async_llm.engine.model_executor.collective_rpc(
            "update_weight", args=(name, dtype, shape, empty_cache)
        )

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.async_llm.engine.model_executor.collective_rpc(
            "update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache)
        )

    def reset_prefix_cache(self):
        self.async_llm.engine.reset_prefix_cache()

    def get_ckp_version(self) -> int:
        return self.ckp_version

    async def sleep(self, level: int = 1):
        await self.async_llm.sleep(level=level)

    async def wake_up(self):
        await self.async_llm.wake_up()
