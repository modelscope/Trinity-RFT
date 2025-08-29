from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from trinity.common.config import FormatConfig, StorageConfig
from trinity.common.constants import PromptType
from trinity.common.experience import Experience
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows import WORKFLOWS, Task
from trinity.utils.registry import Registry

FORMATTER = Registry("formatter")


class ExperienceFormatter(ABC):
    @abstractmethod
    def format(self, sample: Dict) -> Experience:
        """Format a raw sample dict into an experience."""


@FORMATTER.register_module("task")
class TaskFormatter:
    """Formatter for task data.

    Example Input:

    .. code-block:: python

        {
            "input": "Hello",
            "output": "Hi"
        }
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self.is_eval = config.is_eval
        self.default_workflow_cls = WORKFLOWS.get(config.default_workflow_type)  # type: ignore
        if self.is_eval and config.default_eval_workflow_type:
            self.default_workflow_cls = WORKFLOWS.get(config.default_eval_workflow_type)
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(config.default_reward_fn_type)  # type: ignore
        self.workflow_key = config.format.workflow_key
        self.reward_fn_key = config.format.reward_fn_key

    def format(self, sample: Dict) -> Task:
        """Format a raw sample dict into a Task."""

        workflow_name = sample.get(self.workflow_key, None) if self.workflow_key else None
        reward_fn_name = sample.get(self.reward_fn_key, None) if self.reward_fn_key else None

        workflow_cls = (
            WORKFLOWS.get(workflow_name) if workflow_name else None
        ) or self.default_workflow_cls
        reward_fn_cls = (
            REWARD_FUNCTIONS.get(reward_fn_name) if reward_fn_name else None
        ) or self.default_reward_fn_cls
        assert workflow_cls is not None, "`default_workflow_type` or `workflow_key` is required"
        return Task(
            workflow=workflow_cls,
            reward_fn=reward_fn_cls,
            format_args=self.config.format,
            repeat_times=self.config.repeat_times,
            rollout_args=self.config.rollout_args,
            workflow_args=self.config.workflow_args,
            reward_fn_args=self.config.reward_fn_args,
            is_eval=self.is_eval,
            raw_task=sample,
        )


@FORMATTER.register_module("sft")
class SFTFormatter(ExperienceFormatter):
    """Formatter for SFT data, supporting both message list and plaintext formats.

    Uses format_config.prompt_type to distinguish between 'messages' and 'plaintext'.

    Example input of MESSAGES:

    .. code-block:: python

       {
           "messages": [
               {"role": "user", "content": "Hello, how are you?"},
               {"role": "assistant", "content": "I'm fine, thank you!"}
           ]
       }


    Example input of PLAINTEXT:

    .. code-block:: python

        {
            "system_prompt_key": "system",
            "prompt_key": "prompt",
            "response_key": "response",
        }
    """

    def __init__(self, tokenizer, format_config: FormatConfig):
        self.tokenizer = tokenizer
        self.prompt_type = format_config.prompt_type
        # For messages type
        if self.prompt_type == PromptType.MESSAGES:
            self.messages_key = format_config.messages_key
            self.tools_key = format_config.tools_key
        # For plaintext type
        elif self.prompt_type == PromptType.PLAINTEXT:
            self.prompt_key = format_config.prompt_key
            self.response_key = format_config.response_key
            self.system_prompt_key = format_config.system_prompt_key
            self.system_prompt = format_config.system_prompt
            self.tools_key = format_config.tools_key
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def _messages_to_experience(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None
    ) -> Experience:
        tokens = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=False, return_tensors="pt"
        )[0]
        prompt_tokens_ids = self.tokenizer.apply_chat_template(
            messages[:-1], tools=tools, add_generation_prompt=True, return_tensors="pt"
        )[0]
        return Experience(
            tokens=tokens,
            prompt_length=len(prompt_tokens_ids),
            messages=messages,
        )

    def format(self, sample: Dict) -> Experience:
        if self.prompt_type == PromptType.MESSAGES:
            messages = sample[self.messages_key]
        elif self.prompt_type == PromptType.PLAINTEXT:
            messages = []
            if self.system_prompt_key is not None:
                system_message = {"role": "system", "content": sample[self.system_prompt_key]}
                messages.append(system_message)
            elif self.system_prompt is not None:
                system_message = {"role": "system", "content": self.system_prompt}
                messages.append(system_message)
            messages.append({"role": "user", "content": sample[self.prompt_key]})
            messages.append({"role": "assistant", "content": sample[self.response_key]})
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")
        tools = sample.get(self.tools_key, None)
        return self._messages_to_experience(messages, tools)


@FORMATTER.register_module("dpo")
class DPOFormatter(ExperienceFormatter):
    """Formatter for DPO plaintext data.

    Example Input for PLAINTEXT:

    .. code-block:: python

       {
           "prompt": "What is your name?",
           "chosen": "My name is Assistant.",
           "rejected": "I don't have a name."
       }

    Example Input for MESSAGES:

    .. code-block:: python

        {
            "messages": [
                {"role": "user", "content": "What is your name?"},
            ],
            "chosen": [
                {"role": "assistant", "content": "My name is Assistant."},
            ],
            "rejected": [
                {"role": "assistant", "content": "I don't have a favorite color."}
            ]
        }
    """

    def __init__(self, tokenizer, format_config: FormatConfig):
        self.tokenizer = tokenizer
        self.prompt_type = format_config.prompt_type
        if self.prompt_type == PromptType.PLAINTEXT:
            self.prompt_key = format_config.prompt_key
            self.chosen_key = format_config.chosen_key
            self.rejected_key = format_config.rejected_key
            self.system_prompt_key = format_config.system_prompt_key
            self.system_prompt = format_config.system_prompt
        elif self.prompt_type == PromptType.MESSAGES:
            self.messages_key = format_config.messages_key
            self.chosen_key = format_config.chosen_key
            self.rejected_key = format_config.rejected_key
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")
        # currently DPO not support tools

    def _messages_to_experience(
        self, prompt_messages, chosen_messages, rejected_messages
    ) -> Experience:
        prompt_tokens = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, return_tensors="pt"
        )[0]
        chosen_tokens = self.tokenizer.apply_chat_template(
            prompt_messages + chosen_messages, add_generation_prompt=False, return_tensors="pt"
        )[0][len(prompt_tokens) :]
        rejected_tokens = self.tokenizer.apply_chat_template(
            prompt_messages + rejected_messages, add_generation_prompt=False, return_tensors="pt"
        )[0][len(prompt_tokens) :]
        return Experience(
            tokens=prompt_tokens,
            prompt_length=len(prompt_tokens),
            chosen=chosen_tokens,
            rejected=rejected_tokens,
            chosen_messages=prompt_messages + chosen_messages,
            rejected_messages=prompt_messages + rejected_messages,
        )

    def format(self, sample: Dict) -> Experience:
        if self.prompt_type == PromptType.PLAINTEXT:
            messages = []
            if self.system_prompt_key is not None:
                system_message = {"role": "system", "content": sample[self.system_prompt_key]}
                messages.append(system_message)
            elif self.system_prompt is not None:
                system_message = {"role": "system", "content": self.system_prompt}
                messages.append(system_message)
            messages.append({"role": "user", "content": sample[self.prompt_key]})
            chosen = [{"role": "assistant", "content": sample[self.chosen_key]}]
            rejected = [{"role": "assistant", "content": sample[self.rejected_key]}]
        elif self.prompt_type == PromptType.MESSAGES:
            messages = sample[self.messages_key]
            chosen = sample[self.chosen_key]
            rejected = sample[self.rejected_key]
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")
        return self._messages_to_experience(
            prompt_messages=messages,
            chosen_messages=chosen,
            rejected_messages=rejected,
        )
