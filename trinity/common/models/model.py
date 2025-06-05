# -*- coding: utf-8 -*-
"""Base Model Class"""
import socket
import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import openai
import ray
from torch import Tensor

from trinity.common.experience import Experience
from trinity.utils.log import get_logger


class InferenceModel(ABC):
    """A model for high performance for rollout inference."""

    def generate(self, prompts: List[str], **kwargs) -> List[Experience]:
        """Generate a batch of responses from a batch of prompts."""
        raise NotImplementedError

    def chat(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate experiences from a list of history chat messages."""
        raise NotImplementedError

    def logprobs(self, token_ids: List[int]) -> Tensor:
        """Generate logprobs for a list of tokens."""
        raise NotImplementedError

    def convert_messages_to_experience(self, messages: List[dict]) -> Experience:
        """Convert a list of messages into an experience."""
        raise NotImplementedError

    async def generate_async(self, prompt: str, **kwargs) -> List[Experience]:
        """Generate a responses from a prompt in async."""
        raise NotImplementedError

    async def chat_async(self, messages: List[dict], **kwargs) -> List[Experience]:
        """Generate experiences from a list of history chat messages in async."""
        raise NotImplementedError

    async def logprobs_async(self, tokens: List[int]) -> Tensor:
        """Generate logprobs for a list of tokens in async."""
        raise NotImplementedError

    async def convert_messages_to_experience_async(self, messages: List[dict]) -> Experience:
        """Convert a list of messages into an experience in async."""
        raise NotImplementedError

    @abstractmethod
    def get_ckp_version(self) -> int:
        """Get the checkpoint version."""

    def get_available_address(self) -> Tuple[str, int]:
        """Get the address of the actor."""
        address = ray.util.get_node_ip_address()
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return address, port


class ModelWrapper:
    """A wrapper for the InferenceModel Ray Actor"""

    # TODO: check model_type inside __init__
    def __init__(self, model: Any, model_type: str = "vllm"):
        self.model = model
        self.use_async = model_type == "vllm_async"
        self.openai_client: openai.OpenAI = None
        self.logger = get_logger(__name__)

    def generate(self, prompts: List[str], **kwargs) -> List[Experience]:
        if self.use_async:
            results = ray.get(
                [self.model.generate_async.remote(prompt, **kwargs) for prompt in prompts]
            )
            return [exp for exps in results for exp in exps]
        else:
            return ray.get(self.model.generate.remote(prompts, **kwargs))

    def chat(self, messages: List[dict], **kwargs) -> List[Experience]:
        if self.use_async:
            return ray.get(self.model.chat_async.remote(messages, **kwargs))
        else:
            return ray.get(self.model.chat.remote(messages, **kwargs))

    def logprobs(self, tokens: List[int]) -> Tensor:
        if self.use_async:
            return ray.get(self.model.logprobs_async.remote(tokens))
        else:
            return ray.get(self.model.logprobs.remote(tokens))

    def convert_messages_to_experience(self, messages: List[dict]) -> Experience:
        """Convert a list of messages into an experience."""
        if self.use_async:
            return ray.get(self.model.convert_messages_to_experience_async.remote(messages))
        else:
            return ray.get(self.model.convert_messages_to_experience.remote(messages))

    def get_ckp_version(self) -> int:
        return ray.get(self.model.get_ckp_version.remote())

    def get_openai_client(self) -> openai.OpenAI:
        """Get the openai client.

        Returns:
            openai.OpenAI: The openai client. And `model_path` is added to the client which refers to the model path.
        """
        if self.openai_client is not None:
            return self.openai_client
        if not ray.get(self.model.has_api_server.remote()):
            raise ValueError(
                "OpenAI API server is not running on current model."
                "Please set `enable_openai_api` to `True`."
            )
        api_address, model_path = None, None
        while True:
            api_address, model_path = ray.get(self.model.api_server_ready.remote())
            if api_address is not None:
                break
            else:
                self.logger.info("Waiting for OpenAI API server to be ready...")
                time.sleep(5)
        if api_address is None:
            raise RuntimeError(
                "Failed to connect to the API server. Please check the API server is running."
            )
        self.logger.info(f"Successfully connect to API server at {api_address}")
        self.openai_client = openai.OpenAI(
            base_url=api_address,
            api_key="EMPTY",
        )
        setattr(self.openai_client, "model_path", model_path)
        return self.openai_client
