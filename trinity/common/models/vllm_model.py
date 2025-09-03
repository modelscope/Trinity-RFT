"""A wrapper around the vllm.AsyncEngine to handle async requests."""

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import aiohttp
import ray
import torch
import vllm
from transformers import AutoProcessor
from vllm.sampling_params import RequestOutputKind

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.models.mm_utils import (
    attach_images_to_messages,
    build_multi_modal_inputs,
)
from trinity.common.models.model import InferenceModel
from trinity.common.models.utils import get_action_mask_method
from trinity.utils.log import get_logger


# TODO: remove V0 when V1 is stable
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
        self.default_sampling_params = vllm.SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=config.max_response_tokens,
            min_tokens=1,
            truncate_prompt_tokens=config.max_prompt_tokens,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            logprobs=0,
            ignore_eos=config.ignore_eos,
        )
        self.enable_thinking = config.enable_thinking
        self.request_id = 0
        max_model_len = config.max_model_len
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
            disable_log_requests=True,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_chunked_prefill=config.enable_chunked_prefill,
            # max_num_batched_tokens=256, # you can further set this parameter to reduce the vllm peak memory usage
        )
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

    async def _initialize_tokenizer(self):
        if self.tokenizer is None:
            if self.processor and hasattr(self.processor, "tokenizer"):
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = await self.async_llm.get_tokenizer()
        self.tokenizer.truncation_side = "left"

    def _initialize_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

    async def chat(self, messages: List[Dict], **kwargs) -> Sequence[Experience]:
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
        return await self.generate(prompt=prompt, **kwargs)

    async def generate(self, prompt: str, **kwargs) -> Sequence[Experience]:
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
        output = await self._generate_internal(prompt={"prompt_token_ids": token_ids}, **kwargs)
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
        self, messages: List[Dict], raw_mm_data: Dict, **kwargs
    ) -> Sequence[Experience]:
        """Chat with the model with a list of messages in async.

        Args:
            messages (List[dict]): The input history messages.
            raw_mm_data (dict): The raw multi-modal data.
            kwargs (dict): A dictionary of sampling parameters.

        Returns:
            A list of experiences.
        """
        # if self.tokenizer is None:
        #     await self._initialize_tokenizer()
        if self.processor is None:
            self._initialize_processor()
        if self.chat_template is None:
            self.chat_template = self.tokenizer.get_chat_template()
        messages = attach_images_to_messages(messages, raw_mm_data)
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

        mm_inputs = build_multi_modal_inputs(
            prompt=prompt,
            raw_mm_data=raw_mm_data,
            processor=self.processor,
            **kwargs,
        )
        return await self.generate_mm(mm_inputs=mm_inputs, **kwargs)

    async def generate_mm(
        self, prompt: str = None, raw_mm_data: Dict = None, mm_inputs: Dict = None, **kwargs
    ) -> Sequence[Experience]:
        """Generate a response from the provided prompt in async.

        Args:
            prompt (str): The input prompt.
            raw_mm_data (dict): The raw multi-modal data.
            mm_inputs (dict): The multi-modal inputs, already processed.
                - keys: "prompt", "multi_modal_data", "multi_modal_inputs".
            kwargs (dict): A dictionary of sampling parameters.

            Either (`prompt`, raw_mm_data) or (mm_inputs) should be provided.

        Returns:
            A list of experiences.
        """
        if mm_inputs is None:
            mm_inputs = build_multi_modal_inputs(
                prompt=prompt,
                raw_mm_data=raw_mm_data,
                processor=self.processor,
                **kwargs,
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

    async def logprobs(self, token_ids: List[int]) -> torch.Tensor:
        """Calculate the logprobs of the given tokens in async. Please slice the result carefully
        to align with the actual response length.

        Args:
            token_ids (List[int]): The input token ids (seq_length).

        Returns:
            A tensor of logprobs (seq_length - 1).
        """
        output = await self._generate_internal(
            prompt={"prompt_token_ids": token_ids},
            n=1,
            max_tokens=1,
            prompt_logprobs=0,  # vLLM return `prompt_logprobs + 1` logrpobs for each token
        )
        return torch.tensor(
            [list(logprob_dict.values())[0].logprob for logprob_dict in output.prompt_logprobs[1:]],
            dtype=torch.float32,
        )

    async def _generate_internal(self, prompt: Any, **kwargs) -> Any:
        # Send the request to the LLM engine.
        self.request_id += 1
        stream = self.async_llm.generate(
            request_id=str(self.request_id),
            prompt=prompt,
            sampling_params=self._create_sampling_params(**kwargs),
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
    ) -> Experience:
        """Convert a list of messages into an experience."""
        if self.tokenizer is None:
            self.tokenizer = await self.async_llm.get_tokenizer()
        if self.chat_template is None:
            self.chat_template = self.tokenizer.get_chat_template()
        token_ids, action_mask, prompt_length = self.action_mask_method(
            tokenizer=self.tokenizer,
            messages=messages,
            tools=tools,
            chat_template=self.chat_template,
        )  # (seq_length, ), (seq_length, )
        logprobs = await self.logprobs(token_ids=token_ids.tolist())  # (seq_length - 1,)
        return Experience(
            tokens=token_ids,
            logprobs=logprobs[prompt_length - 1 :],
            prompt_length=prompt_length,
            action_mask=action_mask[prompt_length:],  # Exclude the prompt tokens
        )

    def shutdown(self):
        """Shutdown the vLLM v1 engine. This kills child processes forked
        by the vLLM engine. If not called, the child processes will be
        orphaned and will not be killed when the parent process exits,
        and they won't be able to be tracked by Ray anymore.
        """
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

    async def sync_model(self, model_version: int) -> bool:
        """Sync model weights to vLLM."""
        await self._collective_rpc("update_weight")
        self.logger.info("Sync model weights to vLLM successfully.")
        self.model_version = model_version
        return True

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

    async def run_api_server(self):
        """Run the OpenAI API server in a Ray actor.

        Note:
            Do not use `ray.get()` on this method.
            This method will run forever until the server is shut down.
        """
        if not (self.api_server_host is None or self.api_server_port is None):
            raise RuntimeError("API server is already running.")
        from trinity.common.models.api.vllm_patch import run_api_server_in_ray_actor

        self.api_server_host, self.api_server_port = self.get_available_address()
        await run_api_server_in_ray_actor(
            self.async_llm,
            self.api_server_host,
            self.api_server_port,
            self.config.model_path,
            self.config.enable_auto_tool_choice,
            self.config.tool_call_parser,
            self.config.reasoning_parser,
        )

    async def has_api_server(self) -> bool:
        return self.config.enable_openai_api

    async def api_server_ready(self) -> Union[str, None]:
        """Check if the OpenAI API server is ready.

        Returns:
            api_url (str): The URL of the OpenAI API server.
        """
        if not await self.has_api_server():
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.api_server_host}:{self.api_server_port}/health"
                ) as response:
                    if response.status == 200:
                        return f"http://{self.api_server_host}:{self.api_server_port}/v1"
                    else:
                        return None
        except Exception as e:
            self.logger.error(e)
            return None

    async def reset_prefix_cache(self) -> None:
        await self.async_llm.reset_prefix_cache()

    def get_model_version(self) -> int:
        return self.model_version

    async def sleep(self, level: int = 1) -> None:
        await self.async_llm.sleep(level=level)

    async def wake_up(self) -> None:
        await self.async_llm.wake_up()
