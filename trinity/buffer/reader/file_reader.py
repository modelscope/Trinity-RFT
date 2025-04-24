"""Filed based buffer reader."""

from typing import List, Optional

import transformers
from datasets import load_dataset

from trinity.buffer.buffer_reader import BufferReader
from trinity.common.config import BufferConfig, DatasetConfig
from trinity.common.constants import (
    AlgorithmType,
    PromptType,
    ReadStrategy,
    StorageType,
)
from trinity.common.experience import Experience


class FileReader(BufferReader):
    """Reader of the Queue buffer."""

    def __init__(self, meta: DatasetConfig, config: BufferConfig) -> None:
        assert meta.storage_type == StorageType.FILE
        if meta.algorithm_type == AlgorithmType.SFT:
            self.reader = SFTDataReader(meta, config)
        elif meta.algorithm_type == AlgorithmType.DPO:
            self.reader = DPODataReader(meta, config)
        else:
            # TODO: support read rollout task
            raise ValueError(f"Unsupported algorithm type: {meta.algorithm_type}")

    def read(self, strategy: Optional[ReadStrategy] = None) -> List:
        """Read data from the buffer."""
        if strategy is not None and strategy != ReadStrategy.FIFO:
            raise ValueError(f"Unsupported read strategy: {strategy}")
        return self.reader.read()


class SFTDataReader:
    """Reader for SFT file data."""

    def __init__(self, meta: DatasetConfig, config: BufferConfig):
        self.train_split = meta.kwargs.get("train_split", "train")
        self.prompt_type = PromptType(meta.kwargs.get("prompt_type", "messages"))
        self.messages_key = meta.kwargs.get("messages_key", "messages")
        self.prompt_key = meta.kwargs.get("prompt_key", "prompt")
        self.response_key = meta.kwargs.get("response_key", "response")
        self.read_batch_size = config.read_batch_size
        self.dataset = load_dataset(meta.path)[self.train_split]
        self.data_iter = self.dataset.iter(self.read_batch_size, drop_last_batch=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path)

    def read(self) -> List:
        try:
            batch_data = next(self.data_iter)
        except StopIteration:
            self.dataset = self.dataset.shuffle()
            self.data_iter = self.dataset.iter(self.read_batch_size, drop_last_batch=True)
            batch_data = next(self.data_iter)
        exp_list = []
        if self.prompt_type == PromptType.MESSAGES:
            for messages in batch_data[self.messages_key]:
                tokens = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, return_tensors="pt"
                )[0]
                prompt_tokens = self.tokenizer.apply_chat_template(
                    messages[:-1], add_generation_prompt=True, return_tensors="pt"
                )[0]
                experience = Experience(
                    tokens=tokens,
                    prompt_length=len(prompt_tokens),
                )
                exp_list.append(experience)

        elif self.prompt_type == PromptType.CHATPAIR:
            for prompt_messages, response_messages in zip(
                batch_data[self.prompt_key], batch_data[self.response_key]
            ):
                full_messages = prompt_messages + response_messages

                tokens = self.tokenizer.apply_chat_template(
                    full_messages, add_generation_prompt=False, return_tensors="pt"
                )[0]

                prompt_tokens = self.tokenizer.apply_chat_template(
                    prompt_messages, add_generation_prompt=True, return_tensors="pt"
                )[0]

                experience = Experience(
                    tokens=tokens,
                    prompt_length=len(prompt_tokens),
                )
                exp_list.append(experience)

        elif self.prompt_type == PromptType.PLAINTEXT:
            # TODO: support HF format without chat template
            for prompt, response in zip(batch_data[self.prompt_key], batch_data[self.response_key]):
                tokens = self.tokenizer(prompt + response, return_tensors="pt")["input_ids"][0]
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                experience = Experience(
                    tokens=tokens,
                    prompt_length=len(prompt_tokens),
                )
                exp_list.append(experience)
        else:
            raise ValueError(f"Unknown data format: {self.prompt_type}")
        return exp_list


class DPODataReader:
    def __init__(self, meta: DatasetConfig, config: BufferConfig):
        self.train_split = meta.kwargs.get("train_split", "train")
        self.prompt_type = PromptType(meta.kwargs.get("prompt_type", "messages"))
        self.prompt_key = meta.kwargs.get("prompt_key", "prompt")
        self.chosen_key = meta.kwargs.get("chosen_key", "chosen")
        self.rejected_key = meta.kwargs.get("rejected_key", "rejected")
        self.read_batch_size = config.read_batch_size
        self.dataset = load_dataset(meta.path)[self.train_split]
        self.data_iter = self.dataset.iter(self.read_batch_size, drop_last_batch=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path)

    def _get_assistant_message(self, item) -> dict:
        if isinstance(item, List):
            item = item[0]
        if isinstance(item, str):
            return {"role": "assistant", "content": item}
        else:
            return item

    def read(self) -> List:
        try:
            batch_data = next(self.data_iter)
        except StopIteration:
            self.dataset = self.dataset.shuffle()
            self.data_iter = self.dataset.iter(self.read_batch_size, drop_last_batch=True)
            batch_data = next(self.data_iter)
        exp_list = []
        for prompt, chosen, rejected in zip(
            batch_data[self.prompt_key], batch_data[self.chosen_key], batch_data[self.rejected_key]
        ):
            if self.prompt_type == PromptType.MESSAGES:
                prompt_messages = prompt

            elif self.prompt_type == PromptType.PLAINTEXT:
                prompt_messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            else:
                raise ValueError(f"Unknown prompt type: {self.prompt_type}")
            prompt_tokens = self.tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, return_tensors="pt"
            )[0]
            prompt_length = len(prompt_tokens)
            messages_with_chosen = prompt_messages + [self._get_assistant_message(chosen)]
            chosen_tokens = self.tokenizer.apply_chat_template(
                messages_with_chosen,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0][prompt_length:]
            messages_with_rejected = prompt_messages + [self._get_assistant_message(rejected)]
            rejected_tokens = self.tokenizer.apply_chat_template(
                messages_with_rejected,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0][prompt_length:]
            experience = Experience(
                tokens=prompt_tokens,
                prompt_length=len(prompt_tokens),
                chosen=chosen_tokens,
                rejected=rejected_tokens,
            )
            exp_list.append(experience)
        return exp_list
