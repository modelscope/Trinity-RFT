from typing import List, Optional, Sequence

import ray
import tinker
import torch
from tinker import types
from torch import Tensor

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.models.utils import get_action_mask_method
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.log import get_logger


class TinkerModel(InferenceModel):
    def __init__(
        self,
        config: InferenceModelConfig,
    ) -> None:
        self.config = config
        self.model_version = -1
        self.synchronizer = Synchronizer.get_actor(namespace=ray.get_runtime_context().namespace)
        self.logger = get_logger(__name__)
        self.model = None
        self.tokenizer = None
        self.chat_template = None
        if self.config.chat_template:
            self.chat_template = self.config.chat_template
        self.action_mask_method = get_action_mask_method(self.chat_template)
        self.enable_thinking = config.enable_thinking

    async def _initialize_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        trainer_client = await self.service_client.create_lora_training_client_async(
            base_model=self.config.model_path
        )
        self.tokenizer = trainer_client.get_tokenizer()

    async def _generate_internal(self, prompt: dict, **kwargs) -> types.SampleResponse:
        assert self.model is not None
        sampling_params = {
            "max_tokens": kwargs.get("max_tokens", self.config.max_response_tokens),
            "seed": kwargs.get("seed", self.config.seed),
            "temperature": kwargs.get("temperature", 1.0),
            "top_k": kwargs.get("top_k", -1),
            "top_p": kwargs.get("top_p", 1),
        }

        return await self.model.sample_async(
            prompt=types.ModelInput.from_ints(prompt["prompt_token_ids"]),
            sampling_params=sampling_params,
            num_samples=kwargs.get("n", 1),
            include_prompt_logprobs=kwargs.get("include_prompt_logprobs", False),
            topk_prompt_logprobs=kwargs.get("topk_prompt_logprobs", self.config.logprobs),
        )

    async def generate(self, prompt: str, **kwargs) -> Sequence[Experience]:
        """Generate a responses from a prompt in async."""
        if self.tokenizer is None:
            await self._initialize_tokenizer()

        # Tokenize once without truncation to check if truncation is needed
        token_ids = self.tokenizer(  # type: ignore
            prompt,
            truncation=False,
            return_tensors="pt",
        )[
            "input_ids"
        ][0].tolist()

        # Check if truncation is needed and apply it
        if self.config.enable_prompt_truncation and self.config.max_prompt_tokens is not None:
            if len(token_ids) > self.config.max_prompt_tokens:
                self.logger.warning(
                    f"Prompt was truncated to {self.config.max_prompt_tokens} tokens"
                )
                token_ids = token_ids[: self.config.max_prompt_tokens + 1]  # leave one for response
                return [
                    Experience(
                        tokens=token_ids,
                        logprobs=torch.zeros(1, dtype=torch.float32),
                        prompt_length=len(token_ids) - 1,
                        prompt_text=self.tokenizer.decode(token_ids[:-1]),
                        response_text=self.tokenizer.decode(token_ids[-1]),
                        truncate_status="prompt_truncated",
                        reward=0.0,
                    )
                    for _ in range(kwargs.get("n", 1))
                ]

        output = await self._generate_internal(prompt={"prompt_token_ids": token_ids}, **kwargs)
        experiences = [
            Experience(
                tokens=torch.tensor(token_ids + sequence.tokens, dtype=torch.int32),
                logprobs=torch.tensor(sequence.logprobs, dtype=torch.float32),
                prompt_length=len(token_ids),
                prompt_text=self.tokenizer.decode(token_ids),
                response_text=self.tokenizer.decode(sequence.tokens),
            )
            for sequence in output.sequences
        ]

        return experiences

    async def chat(self, messages: List[dict], **kwargs) -> Sequence[Experience]:
        """Generate experiences from a list of history chat messages in async."""
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

    async def logprobs(self, token_ids: List[int], **kwargs) -> Tensor:
        """Generate logprobs for a list of tokens in async."""
        logprobs = await self.model.compute_logprobs_async(types.ModelInput(token_ids))
        return torch.tensor(logprobs[1:], dtype=torch.float32)

    async def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        """Convert a list of messages into an experience in async."""
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

        # Truncate tokens if they exceed the length limit
        assert token_ids is not None
        truncate_status = None
        if self.config.max_model_len is not None and self.config.max_model_len > 0:
            if len(token_ids) > self.config.max_model_len - 1:
                truncate_status = "response_truncated"
                self.logger.warning(
                    f"Warning: {len(token_ids)=} exceeds the length limit {(self.config.max_model_len - 1)=}"
                )
                token_ids = token_ids[: self.config.max_model_len - 1]
                action_mask = action_mask[: self.config.max_model_len - 1]

        temperature = temperature if temperature is not None else self.config.temperature
        logprobs = await self.logprobs(
            token_ids=token_ids.tolist(), temperature=temperature
        )  # (seq_length - 1,)
        return Experience(
            tokens=token_ids,
            logprobs=logprobs[prompt_length - 1 :],
            prompt_length=prompt_length,
            action_mask=action_mask[prompt_length:],  # Exclude the prompt tokens
            messages=messages,
            truncate_status=truncate_status,
        )

    async def prepare(self) -> None:
        """Prepare the model before inference."""
        self.service_client = tinker.ServiceClient()
        self.model = await self.service_client.create_sampling_client_async(
            base_model=self.config.model_path,
        )

    async def sync_model(self, model_version: int) -> int:
        self.model_version = model_version
        remote_sampler_path, _ = await self.synchronizer.get_model_state_dict.remote()
        self.model = await self.service_client.create_sampling_client_async(
            model_path=remote_sampler_path,
        )
        return model_version

    def get_model_version(self) -> int:
        """Get the checkpoint version."""
        return self.model_version

    def get_api_server_url(self) -> Optional[str]:
        """Get the API server URL if available."""
        # TODO: tinker will support openai api later
        return None

    def get_model_path(self) -> Optional[str]:
        """Get the model path"""
        return self.config.model_path  # type: ignore [return-value]
