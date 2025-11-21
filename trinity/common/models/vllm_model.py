"""A wrapper around the vllm.AsyncEngine to handle async requests."""

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import ray
import torch
import vllm
from packaging.version import parse as parse_version
from PIL import Image
from transformers import AutoProcessor
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.models.mm_utils import (
    build_multi_modal_inputs,
    convert_messages_to_mm_format,
)
from trinity.common.models.model import InferenceModel
from trinity.common.models.utils import get_action_mask_method
from trinity.common.models.vllm_patch.api_patch import get_vllm_version
from trinity.utils.log import get_logger


# V0 engine is deprecated since vLLM v0.10.2, related code will be removed in the future.
class vLLMRolloutModel(InferenceModel):
    """Wrapper around the vLLM engine to handle async requests.

    Args:
        config (Config): The config.
    """

    def __init__(
        self,
        config: InferenceModelConfig,
    ) -> None:
        self.logger = get_logger(__name__)
        self.config = config
        self.use_v1 = config.use_v1
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
        if get_vllm_version() >= parse_version("0.11.0"):
            os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        if not config.enforce_eager:
            # To avoid torch compile conflicts when multiple model are started simultaneously.
            # remove this when the following PR is released:
            # https://github.com/vllm-project/vllm/pull/27616
            os.environ["VLLM_CACHE_ROOT"] = os.path.expanduser(
                f"~/.cache/vllm/{config.bundle_indices}"
            )
        self.default_sampling_params = vllm.SamplingParams(
            n=1,
            temperature=config.temperature,
            max_tokens=config.max_response_tokens,
            min_tokens=config.min_response_tokens,
            truncate_prompt_tokens=config.max_prompt_tokens,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            logprobs=config.logprobs,
            ignore_eos=config.ignore_eos,
        )
        self.enable_thinking = config.enable_thinking
        self.request_id = 0
        max_model_len = config.max_model_len
        self.enable_lora = config.enable_lora
        self.default_lora_path = config.lora_kwargs.pop("default_lora_path", None)
        engine_args = vllm.AsyncEngineArgs(
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
            task="generate",
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_chunked_prefill=config.enable_chunked_prefill,
            # max_num_batched_tokens=256, # you can further set this parameter to reduce the vllm peak memory usage
            override_generation_config={  # TODO: find a way to unittest this
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "max_new_tokens": config.max_response_tokens,
            },
            disable_log_stats=True,
            enable_lora=config.enable_lora,
            logprobs_mode="processed_logprobs",
            **config.lora_kwargs,
        )
        if get_vllm_version() > parse_version("0.10.0"):
            engine_args.enable_log_requests = False
        else:
            engine_args.disable_log_requests = True
        if get_vllm_version() >= parse_version("0.11.0"):
            engine_args.reasoning_parser = config.reasoning_parser
        self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.processor = None
        self.tokenizer = None
        self.chat_template = None
        if self.config.chat_template:
            self.chat_template = self.config.chat_template
        self.action_mask_method = get_action_mask_method(self.chat_template)
        self.state_dict_meta = None
        self.model_version = 0  # TODO: resume the value from the checkpoint
        self.api_server_host = None
        self.api_server_port = None
        self.api_server = None
        self.async_lock = asyncio.Lock()

    async def _initialize_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = await self.async_llm.get_tokenizer()
        self.tokenizer.truncation_side = "left"

    def _initialize_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

    async def chat(
        self, messages: List[Dict], lora_request: LoRARequest = None, **kwargs
    ) -> Sequence[Experience]:
        """Chat with the model with a list of messages in async.

        Args:
            messages (List[dict]): The input history messages.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        if self.tokenizer is None:
            await self._initialize_tokenizer()
        if self.chat_template is None:
            self.chat_template = self.tokenizer.get_chat_template()
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
        return await self.generate(prompt=prompt, lora_request=lora_request, **kwargs)

    async def generate(
        self, prompt: str, lora_request: LoRARequest = None, **kwargs
    ) -> Sequence[Experience]:
        """Generate a response from the provided prompt in async.

        Args:
            prompt (str): The input prompt.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        if self.tokenizer is None:
            await self._initialize_tokenizer()
        token_ids = self.tokenizer(  # type: ignore
            prompt, truncation=True, max_length=self.config.max_prompt_tokens, return_tensors="pt"
        )["input_ids"][0].tolist()
        output = await self._generate_internal(
            prompt={"prompt_token_ids": token_ids}, lora_request=lora_request, **kwargs
        )
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
                prompt_text=self.tokenizer.decode(output.prompt_token_ids),
                response_text=output.outputs[i].text,
            )
            for i in range(len(output.outputs))
        ]
        return experiences

    async def chat_mm(
        self, messages: List[Dict], images: List[Image.Image], videos: List[np.ndarray], **kwargs
    ) -> Sequence[Experience]:
        """Chat with the model with a list of messages in async.

        Args:
            messages (List[dict]): The input history messages.
            raw_mm_data (dict): The raw multi-modal data.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        if self.processor is None:
            self._initialize_processor()
        if self.chat_template is None:
            self.chat_template = self.tokenizer.get_chat_template()
        messages = convert_messages_to_mm_format(messages)
        if messages[-1]["role"] == "assistant":
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
                chat_template=self.chat_template,
            )
        else:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template,
                enable_thinking=self.enable_thinking,
            )

        return await self.generate_mm(prompt=prompt, images=images, videos=videos, **kwargs)

    async def generate_mm(
        self,
        prompt: str = None,
        images: List[Image.Image] = None,
        videos: List[np.ndarray] = None,
        **kwargs,
    ) -> Sequence[Experience]:
        """Generate a response from the provided prompt in async.

        Args:
            prompt (str): The input prompt.
            images (List): The list of image inputs.
            videos (List): The list of video inputs.

        Returns:
            A list of experiences.
        """
        mm_inputs = build_multi_modal_inputs(
            prompt=prompt,
            images=images,
            videos=videos,
            processor=self.processor,
        )

        vllm_inputs = {
            "prompt": mm_inputs["prompt"],
            "multi_modal_data": mm_inputs["multi_modal_data"],
        }

        output = await self._generate_internal(prompt=vllm_inputs, **kwargs)
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
                prompt_text=mm_inputs["prompt"],
                response_text=output.outputs[i].text,
                multi_modal_inputs=mm_inputs["multi_modal_inputs"],
            )
            for i in range(len(output.outputs))
        ]
        return experiences

    async def logprobs(  # type: ignore [override]
        self,
        token_ids: List[int],
        lora_request: LoRARequest = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Calculate the logprobs of the given tokens in async. Please slice the result carefully
        to align with the actual response length.

        Args:
            token_ids (List[int]): The input token ids (seq_length). Please make sure the length of
                it does not exceed `max_model_len - 1`.
            lora_request (LoRARequest, optional): The LoRA request. Defaults to None.
            temperature (float): The temperature for scaling logits.

        Returns:
            A tensor of logprobs (seq_length - 1).
        """
        temperature = temperature if temperature is not None else self.config.temperature
        if temperature is None:
            temperature = 1.0
        output = await self._generate_internal(
            prompt={"prompt_token_ids": token_ids},
            lora_request=lora_request,
            n=1,
            max_tokens=1,
            prompt_logprobs=0,  # vLLM return `prompt_logprobs + 1` logrpobs for each token
            temperature=temperature,
        )
        return torch.tensor(
            [list(logprob_dict.values())[0].logprob for logprob_dict in output.prompt_logprobs[1:]],
            dtype=torch.float32,
        )

    async def _generate_internal(
        self, prompt: Any, lora_request: LoRARequest = None, **kwargs
    ) -> Any:
        # Send the request to the LLM engine.
        self.request_id += 1
        stream = self.async_llm.generate(
            request_id=str(self.request_id),
            prompt=prompt,
            sampling_params=self._create_sampling_params(**kwargs),
            lora_request=lora_request,
        )

        # Consume the stream until the request is finished.
        async for request_output in stream:
            if request_output.finished:
                # Bypass the original full prompt.
                # request_output.prompt = request.prompt
                return request_output

        raise RuntimeError("[vLLM] The request is not finished. This should not happen.")

    async def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience."""
        if self.tokenizer is None:
            await self._initialize_tokenizer()
        if self.chat_template is None:
            self.chat_template = self.tokenizer.get_chat_template()
        token_ids, action_mask, prompt_length = self.action_mask_method(
            tokenizer=self.tokenizer,
            messages=messages,
            tools=tools,
            chat_template=self.chat_template,
            enable_thinking=self.enable_thinking,
        )  # (seq_length, ), (seq_length, )
        temperature = temperature if temperature is not None else self.config.temperature
        logprobs = await self.logprobs(
            token_ids=token_ids.tolist(), temperature=temperature
        )  # (seq_length - 1,)
        return Experience(
            tokens=token_ids,
            logprobs=logprobs[prompt_length - 1 :],
            prompt_length=prompt_length,
            action_mask=action_mask[prompt_length:],  # Exclude the prompt tokens
        )

    async def shutdown(self):
        """Shutdown the vLLM v1 engine. This kills child processes forked
        by the vLLM engine. If not called, the child processes will be
        orphaned and will not be killed when the parent process exits,
        and they won't be able to be tracked by Ray anymore.
        """
        if self.api_server is not None:
            self.api_server.cancel()
            try:
                await self.api_server
            except asyncio.CancelledError:
                pass
            self.api_server = None
        if hasattr(self.async_llm, "shutdown"):
            self.logger.info("Shutting down vLLM engine")
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

    async def _collective_rpc(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        if self.use_v1:
            return await self.async_llm.collective_rpc(method, timeout, args, kwargs)
        else:
            return self.async_llm.engine.model_executor.collective_rpc(
                method, timeout, args, kwargs
            )

    async def sync_model(self, model_version: int) -> int:
        """Sync model weights to vLLM."""
        if self.enable_lora:
            # Revise the lora path; no need to sync weights manually.
            self.default_lora_path = self.default_lora_path.replace(
                f"global_step_{self.model_version}", f"global_step_{model_version}"
            )
            self.logger.info(
                f"Redirect `lora_path` from old_model_version={self.model_version} to {model_version=} successfully."
            )
            lora_int_ids = await self.async_llm.list_loras()
            for lora_id in lora_int_ids:
                await self.async_llm.remove_lora(lora_id)
            await self.async_llm.add_lora(self.get_lora_request(self.default_lora_path))
            self.model_version = model_version
            return model_version
        await self._collective_rpc("update_weight")
        self.logger.info("Sync model weights to vLLM successfully.")
        self.model_version = model_version
        return model_version

    async def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        explorer_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        state_dict_meta: dict = None,
    ):
        return await self._collective_rpc(
            "init_process_group",
            args=(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
                timeout,
                state_dict_meta,
                explorer_name,
                ray.get_runtime_context().namespace,
            ),
        )

    async def run_api_server(self) -> bool:
        """Run the OpenAI API server in a Ray actor.

        Returns:
            success (bool): Whether the API server is started successfully.
        """
        async with self.async_lock:
            if not self.config.enable_openai_api:
                return False  # Not enabled

            if self.api_server_host is not None and self.api_server_port is not None:
                return True  # already running

            from trinity.common.models.vllm_patch.api_patch import (
                run_api_server_in_ray_actor,
            )

            api_server_host, api_server_port = self.get_available_address()
            self.api_server = asyncio.create_task(
                run_api_server_in_ray_actor(
                    self.async_llm,
                    api_server_host,
                    api_server_port,
                    self.config.model_path,
                    self.config.enable_auto_tool_choice,
                    self.config.tool_call_parser,
                    self.config.reasoning_parser,
                )
            )
            self.api_server_host = api_server_host
            self.api_server_port = api_server_port
            return True

    async def get_api_server_url(self) -> Optional[str]:
        """Get the URL of the OpenAI API server.

        Returns:
            api_url (str): The URL of the OpenAI API server.
        """
        if not await self.run_api_server():
            return None
        return f"http://{self.api_server_host}:{self.api_server_port}"

    async def reset_prefix_cache(self) -> None:
        await self.async_llm.reset_prefix_cache()

    def get_model_version(self) -> int:
        return self.model_version

    def get_model_path(self) -> str:
        return self.config.model_path

    def get_lora_request(self, lora_path: Optional[str] = None) -> LoRARequest:
        assert self.config.lora_modules is not None
        lora_request = LoRARequest(**self.config.lora_modules[0])
        if lora_path is not None:
            self.config.lora_modules[0]["lora_path"] = lora_path  # for consistency
            lora_request.lora_path = lora_path
        return lora_request

    async def sleep(self, level: int = 1) -> None:
        await self.async_llm.sleep(level=level)

    async def wake_up(self) -> None:
        await self.async_llm.wake_up()
