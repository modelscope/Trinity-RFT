from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from trinity.common.config import FormatConfig
from trinity.common.experience import Experience


class Formatter(ABC):
    @abstractmethod
    def __init__(self, format_config: FormatConfig):
        """Initialize the formatter with the given configuration."""

    @abstractmethod
    def format(self, sample: Dict) -> Experience:
        """Format a raw sample dict into an experience."""


def format_messages(
    tokenizer, messages: List[Dict], tools: Optional[List[Dict]] = None
) -> Experience:
    tokens = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=False, return_tensors="pt"
    )[0]
    prompt_tokens_ids = tokenizer.apply_chat_template(
        messages[:-1], tools=tools, add_generation_prompt=True, return_tensors="pt"
    )[0]
    return Experience(
        tokens=tokens,
        prompt_length=len(prompt_tokens_ids),
    )


class SFTMessagesFormatter(Formatter):
    """Formatter for SFT message list data.

    Example Input:

    .. code-block:: python

       {
           "messages": [
               {"role": "user", "content": "Hello, how are you?"},
               {"role": "assistant", "content": "I'm fine, thank you!"}
           ]
       }
    """

    def __init__(self, tokenizer, format_config: FormatConfig):
        self.tokenizer = tokenizer
        self.messages_key = format_config.messages_key
        self.tools_key = format_config.tools_key

    def format(self, sample: Dict) -> Experience:
        """Format a raw sample dict into an experience."""
        messages = sample[self.messages_key]
        tools = sample.get(self.tools_key, None)

        return format_messages(self.tokenizer, messages, tools)


class SFTPlaintextFormatter(Formatter):
    """Formatter for SFT plaintext data.

    Example Input:

    .. code-block:: python

        {
            "system_prompt_key": "system",
            "prompt_key": "prompt",
            "response_key": "response",
        }
    """

    def __init__(self, tokenizer, format_config: FormatConfig):
        self.tokenizer = tokenizer
        self.prompt_key = format_config.prompt_key
        self.response_key = format_config.response_key
        self.system_prompt_key = format_config.system_prompt_key
        self.system_prompt = format_config.system_prompt
        self.tools_key = format_config.tools_key

    def format(self, sample: Dict) -> Experience:
        """Format a raw sample dict into an experience."""
        messages = []
        if self.system_prompt_key is not None:
            system_message = {"role": "system", "content": sample[self.system_prompt_key]}
            messages.append(system_message)
        elif self.system_prompt is not None:
            system_message = {"role": "system", "content": self.system_prompt}
            messages.append(system_message)
        messages.append({"role": "user", "content": sample[self.prompt_key]})
        messages.append({"role": "assistant", "content": sample[self.response_key]})

        return format_messages(self.tokenizer, messages, sample.get(self.tools_key, None))


def format_dpo_messages(
    tokenizer,
    prompt_messages: List[Dict],
    chosen_messages: List[Dict],
    reject_messages: List[Dict],
):
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, return_tensors="pt"
    )[0]
    chosen_tokens = tokenizer.apply_chat_template(
        prompt_messages + chosen_messages, add_generation_prompt=False, return_tensors="pt"
    )[0][len(prompt_tokens) :]
    reject_tokens = tokenizer.apply_chat_template(
        prompt_messages + reject_messages, add_generation_prompt=False, return_tensors="pt"
    )[0][len(prompt_tokens) :]
    return Experience(
        tokens=prompt_tokens,
        prompt_length=len(prompt_tokens),
        chosen=chosen_tokens,
        rejected=reject_tokens,
    )


class DPOPlaintextFormatter(Formatter):
    """Formatter for DPO plaintext data.

    Example Input:

    .. code-block:: python

       {
           "prompt": "What is your name?",
           "chosen": "My name is Assistant.",
           "rejected": "I don't have a name."
       }
    """

    def __init__(self, tokenizer, format_config: FormatConfig):
        self.tokenizer = tokenizer
        self.prompt_key = format_config.prompt_key
        self.chosen_key = format_config.chosen_key
        self.rejected_key = format_config.rejected_key
        self.system_prompt_key = format_config.system_prompt_key
        self.system_prompt = format_config.system_prompt
        # currently DPO not support tools

    def format(self, sample: Dict) -> Experience:
        messages = []
        if self.system_prompt_key is not None:
            system_message = {"role": "system", "content": sample[self.system_prompt_key]}
            messages.append(system_message)
        elif self.system_prompt is not None:
            system_message = {"role": "system", "content": self.system_prompt}
            messages.append(system_message)
        messages.append({"role": "user", "content": sample[self.prompt_key]})
        return format_dpo_messages(
            tokenizer=self.tokenizer,
            prompt_messages=messages,
            chosen_messages=[{"role": "assistant", "content": sample[self.chosen_key]}],
            reject_messages=[{"role": "assistant", "content": sample[self.rejected_key]}],
        )


class DPOMessagesFormatter(Formatter):
    """Formatter for DPO message list data.

    Example Input:

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
        self.messages_key = format_config.messages_key
        self.chosen_key = format_config.chosen_key
        self.rejected_key = format_config.rejected_key

    def format(self, sample: Dict) -> Experience:
        messages = sample[self.messages_key]
        chosen = sample[self.chosen_key]
        rejected = sample[self.rejected_key]
        return format_dpo_messages(self.tokenizer, messages, chosen, rejected)
