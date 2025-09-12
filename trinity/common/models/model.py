# -*- coding: utf-8 -*-
"""Base Model Class"""
import asyncio
import socket
import time
from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple, Union

import openai
import ray
import torch
from torch import Tensor

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

    async def logprobs(self, tokens: List[int]) -> Tensor:
        """Generate logprobs for a list of tokens in async."""
        raise NotImplementedError

    async def convert_messages_to_experience(self, messages: List[dict]) -> Experience:
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

    def __init__(self, model: Any, model_type: str = "vllm", enable_history: bool = False):
        assert model_type.startswith("vllm"), "Only vLLM model is supported for now."
        self.model = model
        self.api_address: str = None
        self.openai_client: openai.OpenAI = None
        self.openai_async_client: openai.AsyncOpenAI = None
        self.logger = get_logger(__name__)
        self.enable_history = enable_history
        self.history = []

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
        results = ray.get([self.model.generate.remote(prompt, **kwargs) for prompt in prompts])
        return [exp for exps in results for exp in exps]

    @_history_recorder
    async def generate_async(self, prompts: List[str], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of prompts in async."""
        results = await asyncio.gather(
            *[self.model.generate.remote(prompt, **kwargs) for prompt in prompts]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    def generate_mm(
        self, prompts: List[str], raw_mm_data_list: List[dict], **kwargs
    ) -> List[Experience]:
        """Generate a list of experiences from a list of prompts and raw_mm_data."""
        results = ray.get(
            [
                self.model.generate_mm.remote(prompt, mm_data, **kwargs)
                for prompt, mm_data in zip(prompts, raw_mm_data_list)
            ]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    async def generate_mm_async(
        self, prompts: List[str], raw_mm_data_list: List[dict], **kwargs
    ) -> List[Experience]:
        results = await asyncio.gather(
            *[
                self.model.generate_mm.remote(p, m, **kwargs)
                for p, m in zip(prompts, raw_mm_data_list)
            ]
        )
        return [exp for exps in results for exp in exps]

    @_history_recorder
    def chat(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of messages."""
        return ray.get(self.model.chat.remote(messages, **kwargs))

    @_history_recorder
    async def chat_async(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate a list of experiences from a list of messages in async."""
        return await self.model.chat.remote(messages, **kwargs)

    @_history_recorder
    def chat_mm(self, messages: List[dict], raw_mm_data: dict, **kwargs) -> List[Experience]:
        return ray.get(self.model.chat_mm.remote(messages, raw_mm_data, **kwargs))

    @_history_recorder
    async def chat_mm_async(
        self, messages: List[dict], raw_mm_data: dict, **kwargs
    ) -> List[Experience]:
        return await self.model.chat_mm.remote(messages, raw_mm_data, **kwargs)

    def logprobs(self, tokens: List[int]) -> Tensor:
        """Calculate the logprobs of the given tokens."""
        return ray.get(self.model.logprobs.remote(tokens))

    async def logprobs_async(self, tokens: List[int]) -> Tensor:
        """Calculate the logprobs of the given tokens in async."""
        return await self.model.logprobs.remote(tokens)

    def convert_messages_to_experience(self, messages: List[dict]) -> Experience:
        """Convert a list of messages into an experience."""
        return ray.get(self.model.convert_messages_to_experience.remote(messages))

    async def convert_messages_to_experience_async(self, messages: List[dict]) -> Experience:
        """Convert a list of messages into an experience in async."""
        return await self.model.convert_messages_to_experience.remote(messages)

    @property
    def model_version(self) -> int:
        """Get the version of the model."""
        return ray.get(self.model.get_model_version.remote())

    def _get_api_server_address(self) -> str:
        """Get the address of the API server."""
        if self.api_address:
            return self.api_address
        if not ray.get(self.model.has_api_server.remote()):
            raise ValueError(
                "OpenAI API server is not running on current model."
                "Please set `enable_openai_api` to `True`."
            )
        api_address = None
        while True:
            api_address = ray.get(self.model.api_server_ready.remote())
            if api_address is not None:
                break
            else:
                self.logger.info("Waiting for OpenAI API server to be ready...")
                time.sleep(5)
        if api_address is None:
            raise RuntimeError(
                "Failed to connect to the API server. Please check the API server is running."
            )
        self.api_address = api_address
        self.logger.info(f"Successfully connect to API server at {api_address}")
        return api_address

    def get_openai_client(self) -> openai.OpenAI:
        """Get the openai client.

        Returns:
            openai.OpenAI: The openai client. And `model_path` is added to the client which refers to the model path.
        """
        if self.openai_client is not None:
            return self.openai_client
        api_address = self._get_api_server_address()
        self.openai_client = openai.OpenAI(
            base_url=api_address,
            api_key="EMPTY",
        )
        if self.enable_history:
            # add a decorator to the openai client to record history
            ori_create = self.openai_client.chat.completions.create

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
        # first make sure that we have the sync openai client
        api_address = self._get_api_server_address()
        self.openai_async_client = openai.AsyncOpenAI(
            base_url=api_address,
            api_key="EMPTY",
        )
        if self.enable_history:
            # add a decorator to the openai client to record history
            ori_create = self.openai_async_client.chat.completions.create

            async def record_chat_completions(*args, **kwargs):
                response = await ori_create(*args, **kwargs)
                self.history.extend(convert_api_output_to_experience(response))
                return response

            self.openai_async_client.chat.completions.create = record_chat_completions
        # get model_path from the sync openai client to avoid async call here
        openai_client = self.get_openai_client()
        setattr(self.openai_async_client, "model_path", openai_client.models.list().data[0].id)
        return self.openai_async_client

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
