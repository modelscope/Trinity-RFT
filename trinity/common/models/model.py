# -*- coding: utf-8 -*-
"""Base Model Class"""
import asyncio
import socket
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import httpx
import numpy as np
import openai
import ray
import torch
from PIL import Image
from torch import Tensor
from vllm.lora.request import LoRARequest

from trinity.common.constants import RunningStatus
from trinity.common.experience import Experience
from trinity.utils.log import get_logger


class InferenceModel(ABC):
    """A model for high performance for rollout inference."""

    async def generate(self, prompt: str, **kwargs) -> Sequence[Experience]:
        """Generate a responses from a prompt in async."""
        raise NotImplementedError

    async def chat(self, messages: List[dict], **kwargs) -> Sequence[Experience]:
        """Generate experiences from a list of history chat messages in async."""
        raise NotImplementedError

    async def logprobs(self, token_ids: List[int], **kwargs) -> Tensor:
        """Generate logprobs for a list of tokens in async."""
        raise NotImplementedError

    async def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience in async."""
        raise NotImplementedError

    @abstractmethod
    def get_model_version(self) -> int:
        """Get the checkpoint version."""

    def get_available_address(self) -> Tuple[str, int]:
        """Get the address of the actor."""
        address = ray.util.get_node_ip_address()
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return address, port

    async def get_api_server_url(self) -> Optional[str]:
        """Get the API server URL if available."""
        return None

    def get_model_path(self) -> Optional[str]:
        """Get the model path"""
        return None


def _history_recorder(func):
    """Decorator to record history of the model calls."""

    async def async_wrapper(self, *args, **kwargs):
        result = await func(self, *args, **kwargs)
        if self.enable_history:
            self._record_history(result)
        return result

    def sync_wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.enable_history:
            self._record_history(result)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class ModelWrapper:
    """A wrapper for the InferenceModel Ray Actor"""

    def __init__(
        self,
        model: InferenceModel,
        engine_type: str = "vllm",
        enable_lora: bool = False,
        enable_history: bool = False,
    ):
        assert engine_type.startswith("vllm"), "Only vLLM model is supported for now."
        self.model = model
        self.api_address: str = None
        self.openai_client: openai.OpenAI = None
        self.openai_async_client: openai.AsyncOpenAI = None
        self.logger = get_logger(__name__)
        self.enable_lora = enable_lora
        self.enable_history = enable_history
        self.history = []
        self.status = RunningStatus.RUNNING
        self.request_count = 0

    async def prepare(self) -> None:
        """Prepare the model wrapper."""
        self.api_address = await self.model.get_api_server_url.remote()
        if self.api_address is None:
            self.logger.info("API server is not enabled for inference model.")
            return
        max_retries = 30
        interval = 2  # seconds
        for i in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.api_address + "/health", timeout=5)
                    if response.status_code == 200:
                        return
            except Exception as e:
                self.logger.info(f"API server not ready (attempt {i + 1}/{max_retries}): {e}")
            await asyncio.sleep(interval)
        raise RuntimeError(
            f"API server at {self.api_address} not ready after {max_retries} attempts."
        )

    def _record_history(self, exps: Union[Experience, List[Experience]]) -> None:
        """Record experiences to history."""
        if isinstance(exps, Experience):
            self.history.append(exps)
        elif isinstance(exps, list):
            self.history.extend(exps)
        else:
            raise TypeError("Expected Experience or List[Experience], got {}".format(type(exps)))

    @_history_recorder
    def generate(self, prompts: List[str], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of prompts."""
        lora_request = self.get_lora_request()
        results = ray.get(
            [self.model.generate.remote(prompt, lora_request, **kwargs) for prompt in prompts]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    async def generate_async(self, prompts: List[str], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of prompts in async."""
        lora_request = await self.get_lora_request_async()
        results = await asyncio.gather(
            *[self.model.generate.remote(prompt, lora_request, **kwargs) for prompt in prompts]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    def generate_mm(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        videos: List[List[np.ndarray]],
        **kwargs,
    ) -> List[Experience]:
        """Generate a list of experiences from a list of prompts and multi-modal data."""
        results = ray.get(
            [
                self.model.generate_mm.remote(prompt, images=img, videos=vid, **kwargs)
                for prompt, img, vid in zip(prompts, images, videos)
            ]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    async def generate_mm_async(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        videos: List[List[np.ndarray]],
        **kwargs,
    ) -> List[Experience]:
        results = await asyncio.gather(
            *[
                self.model.generate_mm.remote(p, images=img, videos=vid, **kwargs)
                for p, img, vid in zip(prompts, images, videos)
            ]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    def chat(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of messages."""
        lora_request = self.get_lora_request()
        return ray.get(self.model.chat.remote(messages, lora_request, **kwargs))

    @_history_recorder
    async def chat_async(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of messages in async."""
        lora_request = await self.get_lora_request_async()
        return await self.model.chat.remote(messages, lora_request, **kwargs)

    @_history_recorder
    def chat_mm(
        self, messages: List[dict], images: List[Image.Image], videos: List[np.ndarray], **kwargs
    ) -> List[Experience]:
        return ray.get(self.model.chat_mm.remote(messages, images=images, videos=videos, **kwargs))

    @_history_recorder
    async def chat_mm_async(
        self, messages: List[dict], images: List[Image.Image], videos: List[np.ndarray], **kwargs
    ) -> List[Experience]:
        return await self.model.chat_mm.remote(messages, images=images, videos=videos, **kwargs)

    def logprobs(self, tokens: List[int], temperature: Optional[float] = None) -> Tensor:
        """Calculate the logprobs of the given tokens."""
        return ray.get(self.model.logprobs.remote(tokens, temperature=temperature))

    async def logprobs_async(
        self, tokens: List[int], temperature: Optional[float] = None
    ) -> Tensor:
        """Calculate the logprobs of the given tokens in async."""
        return await self.model.logprobs.remote(tokens, temperature=temperature)

    def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience."""
        return ray.get(
            self.model.convert_messages_to_experience.remote(
                messages, tools=tools, temperature=temperature
            )
        )

    async def convert_messages_to_experience_async(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience in async."""
        return await self.model.convert_messages_to_experience.remote(
            messages, tools=tools, temperature=temperature
        )

    @property
    def model_version(self) -> int:
        """Get the version of the model."""
        return ray.get(self.model.get_model_version.remote())

    @property
    async def model_version_async(self) -> int:
        """Get the version of the model."""
        return await self.model.get_model_version.remote()

    @property
    def model_path(self) -> str:
        """Get the model path."""
        return ray.get(self.model.get_model_path.remote())

    @property
    async def model_path_async(self) -> str:
        """Get the model path."""
        return await self.model.get_model_path.remote()

    def get_lora_request(self) -> Optional[LoRARequest]:
        if self.enable_lora:
            return ray.get(self.model.get_lora_request.remote())
        else:
            return None

    async def get_lora_request_async(self) -> Optional[LoRARequest]:
        if self.enable_lora:
            return await self.model.get_lora_request.remote()
        else:
            return None

    def get_openai_client(self) -> openai.OpenAI:
        """Get the openai client.

        Returns:
            openai.OpenAI: The openai client. And `model_path` is added to the client which refers to the model path.
        """
        if self.openai_client is not None:
            return self.openai_client
        if not self.api_address:
            raise ValueError(
                "API server is not enabled for this model. OpenAI client is unavailable."
            )
        self.openai_client = openai.OpenAI(
            base_url=f"{self.api_address}/v1",
            api_key="EMPTY",
        )
        if self.enable_history:
            # add a decorator to the openai client to record history
            ori_create = partial(self.openai_client.chat.completions.create, logprobs=True)

            def record_chat_completions(*args, **kwargs):
                response = ori_create(*args, **kwargs)
                self.history.extend(convert_api_output_to_experience(response))
                return response

            self.openai_client.chat.completions.create = record_chat_completions
        setattr(self.openai_client, "model_path", self.openai_client.models.list().data[0].id)
        return self.openai_client

    def get_openai_async_client(self) -> openai.AsyncOpenAI:
        """Get the async openai client.

        Returns:
            openai.AsyncOpenAI: The async openai client. And `model_path` is added to the client which refers to the model path.
        """
        if self.openai_async_client is not None:
            return self.openai_async_client
        if not self.api_address:
            raise ValueError(
                "API server is not enabled for this model. OpenAI async client is unavailable."
            )
        # first make sure that we have the sync openai client
        self.openai_async_client = openai.AsyncOpenAI(
            base_url=f"{self.api_address}/v1",
            api_key="EMPTY",
        )
        if self.enable_history:
            # add a decorator to the openai client to record history
            ori_create = partial(self.openai_async_client.chat.completions.create, logprobs=True)

            async def record_chat_completions(*args, **kwargs):
                response = await ori_create(*args, **kwargs)
                self.history.extend(convert_api_output_to_experience(response))
                return response

            self.openai_async_client.chat.completions.create = record_chat_completions
        # get model_path from the sync openai client to avoid async call here
        openai_client = self.get_openai_client()
        setattr(self.openai_async_client, "model_path", openai_client.models.list().data[0].id)
        return self.openai_async_client

    async def get_current_load(self) -> int:
        """Get the current load metrics of the model."""
        if not self.api_address:
            raise ValueError(
                "API server is not enabled for this model. Load metrics is unavailable."
            )
        with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_address}/load")
            data = response.json()
            return data["server_load"]

    async def sync_model_weights(self, model_version: int) -> None:
        """Sync the model weights"""
        await self.model.sync_model.remote(model_version)

    def extract_experience_from_history(self, clear_history: bool = True) -> List[Experience]:
        """Extract experiences from the history."""
        if not self.enable_history:
            raise ValueError("History recording is not enabled.")
        exps = [exp for exp in self.history]
        if clear_history:
            self.history.clear()
        return exps


def convert_api_output_to_experience(
    output,
) -> List[Experience]:
    """Convert the API output to a list of experiences."""
    return [
        Experience(
            tokens=torch.cat(
                (
                    torch.tensor(output.prompt_token_ids, dtype=torch.int32),
                    torch.tensor(choice.token_ids, dtype=torch.int32),
                )
            ),
            logprobs=extract_logprobs(choice),
            prompt_length=len(output.prompt_token_ids),
            response_text=choice.message.content,
        )
        for choice in output.choices
    ]


def extract_logprobs(choice) -> Tensor:
    """Extract logprobs from a list of logprob dictionaries."""
    if not hasattr(choice, "logprobs") or choice.logprobs is None:
        return torch.tensor([], dtype=torch.float32)
    return torch.tensor(
        [logprob.logprob for logprob in choice.logprobs.content],
        dtype=torch.float32,
    )
