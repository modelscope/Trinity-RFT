"""Filed based buffer reader."""

import copy
from typing import List, Optional

import datasets
import transformers
from datasets import Dataset, load_dataset

from trinity.algorithm.algorithm import DPOAlgorithm, SFTAlgorithm
from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.schema.formatter import (
    DPOMessagesFormatter,
    DPOPlaintextFormatter,
    SFTMessagesFormatter,
    SFTPlaintextFormatter,
)
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import PromptType, TaskType
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows import WORKFLOWS, Task
from trinity.utils.registry import Registry

FILE_READERS = Registry("file_readers")


class DummyProgressBar:
    def __init__(self):
        pass

    def update(self, num: int):
        pass

    def close(self):
        pass


class _HFBatchReader:
    def __init__(
        self,
        dataset: Dataset,
        name: str,
        default_batch_size: int,
        total_epochs: int = 1,
        offset: int = 0,
        drop_last: bool = True,
        total_steps: Optional[int] = None,
        enable_progress_bar: Optional[bool] = True,
    ):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.name = name
        self.current_batch_size = None
        self.drop_last = drop_last

        self.current_offset = offset
        self.iter = iter(self.dataset)

        for _ in range(self.current_offset % self.dataset_size):
            next(self.iter)

        # convert epochs/steps to sample number
        if total_steps:
            self.total_samples = default_batch_size * total_steps
        else:
            self.total_samples = self.dataset_size * total_epochs

        if enable_progress_bar:
            from ray.experimental.tqdm_ray import tqdm

            self.progress_bar = tqdm(
                total=self.total_samples,
                desc=f"Dataset [{self.name}] Progressing",
            )
        else:
            self.progress_bar = DummyProgressBar()

        self.progress_bar.update(self.current_offset)

    def read_batch(self, batch_size: int) -> List:
        if self.current_offset >= self.total_samples:
            self.progress_bar.close()
            raise StopIteration
        batch = []

        while len(batch) < batch_size:
            try:
                item = next(self.iter)
                batch.append(item)
                self.current_offset += 1
            except StopIteration:
                if self.current_offset >= self.total_samples:
                    # No more data to read
                    if not self.drop_last and len(batch) > 0:
                        # return last batch
                        self.progress_bar.update(len(batch))
                        return batch
                    else:
                        self.progress_bar.close()
                        raise StopIteration
                # Step to the next epoch
                self.iter = iter(self.dataset)
        self.progress_bar.update(batch_size)
        return batch


class BaseFileReader(BufferReader):
    async def read_async(self, batch_size: Optional[int] = None):
        try:
            return self.read(batch_size)
        except StopIteration as e:
            raise StopAsyncIteration from e


@FILE_READERS.register_module(SFTAlgorithm.name())
class SFTDataReader(BaseFileReader):
    """Reader for SFT file data."""

    def __init__(self, meta: StorageConfig, config: BufferConfig):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path)
        if meta.format.prompt_type == PromptType.MESSAGES:
            self.formatter = SFTMessagesFormatter(
                tokenizer=self.tokenizer, format_config=meta.format
            )
        elif meta.format.prompt_type == PromptType.PLAINTEXT:
            self.formatter = SFTPlaintextFormatter(
                tokenizer=self.tokenizer, format_config=meta.format
            )
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")
        self.read_batch_size = config.train_batch_size
        self.dataset = _HFBatchReader(
            load_dataset(meta.path, name=meta.subset_name, split=meta.split),
            name=meta.name,
            default_batch_size=self.read_batch_size,
            total_epochs=meta.total_epochs,
            drop_last=True,
            total_steps=meta.total_steps,
            enable_progress_bar=meta.enable_progress_bar,
        )

    def read(self, batch_size: Optional[int] = None) -> List:
        samples = self.dataset.read_batch(batch_size or self.read_batch_size)
        exp_list = []
        for sample in samples:
            experience = self.formatter.format(sample)
            exp_list.append(experience)
        return exp_list


@FILE_READERS.register_module(DPOAlgorithm.name())
class DPODataReader(BaseFileReader):
    def __init__(self, meta: StorageConfig, config: BufferConfig):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path)
        if meta.format.prompt_type == PromptType.MESSAGES:
            self.formatter = DPOMessagesFormatter(
                tokenizer=self.tokenizer, format_config=meta.format
            )
        elif meta.format.prompt_type == PromptType.PLAINTEXT:
            self.formatter = DPOPlaintextFormatter(
                tokenizer=self.tokenizer, format_config=meta.format
            )
        self.read_batch_size = config.train_batch_size
        self.dataset = _HFBatchReader(
            load_dataset(meta.path, name=meta.subset_name, split=meta.split),
            name=meta.name,
            default_batch_size=self.read_batch_size,
            total_epochs=meta.total_epochs,
            drop_last=True,
            total_steps=meta.total_steps,
            enable_progress_bar=meta.enable_progress_bar,
        )  # TODO: support resume

    def _get_assistant_message(self, item) -> dict:
        if isinstance(item, List):
            item = item[0]
        if isinstance(item, str):
            return {"role": "assistant", "content": item}
        else:
            return item

    def read(self, batch_size: Optional[int] = None) -> List:
        batch_data = self.dataset.read_batch(batch_size or self.read_batch_size)
        exp_list = []
        for sample in batch_data:
            experience = self.formatter.format(sample)
            exp_list.append(experience)
        return exp_list


@FILE_READERS.register_module("rollout")
class RolloutDataReader(BaseFileReader):
    def __init__(self, meta: StorageConfig, config: BufferConfig):
        self.meta = meta
        self.name = meta.name
        self.split = meta.split
        subset_name = meta.subset_name
        # disable datasets caching to avoid reuse old-version dataset
        self.epoch = 0
        datasets.disable_caching()
        self.read_batch_size = config.batch_size
        self.dataset = _HFBatchReader(
            load_dataset(meta.path, name=subset_name, split=self.split),
            name=meta.name,
            default_batch_size=self.read_batch_size,
            total_epochs=self.meta.total_epochs if meta.task_type == TaskType.EXPLORE else 1,
            offset=self.meta.index,
            drop_last=self.meta.task_type == TaskType.EXPLORE,
            total_steps=meta.total_steps,
            enable_progress_bar=meta.enable_progress_bar,
        )
        self.prompt_key = meta.format.prompt_key
        self.response_key = meta.format.response_key
        self.image_key = meta.format.image_key
        self.video_key = meta.format.video_key
        self.workflow_key = meta.format.workflow_key
        self.reward_fn_key = meta.format.reward_fn_key

        self.task_type = meta.task_type
        self.default_workflow_cls = WORKFLOWS.get(meta.default_workflow_type)  # type: ignore
        self.default_eval_workflow_cls = None
        if getattr(meta, "default_eval_workflow_type", None):
            self.default_eval_workflow_cls = WORKFLOWS.get(meta.default_eval_workflow_type)
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(meta.default_reward_fn_type)  # type: ignore

    def read(self, batch_size: Optional[int] = None) -> List:
        batch_size = batch_size or self.read_batch_size
        tasks = []
        samples = self.dataset.read_batch(batch_size)
        for sample in samples:
            if self.task_type == TaskType.EVAL and self.default_eval_workflow_cls:
                workflow_class = self.default_eval_workflow_cls
            else:
                workflow_class = (
                    WORKFLOWS.get(sample[self.workflow_key])
                    if self.workflow_key in sample
                    else self.default_workflow_cls
                )
            reward_fn = (
                REWARD_FUNCTIONS.get(sample[self.reward_fn_key])
                if self.reward_fn_key in sample
                else self.default_reward_fn_cls
            )
            assert (
                workflow_class is not None
            ), "`default_workflow_type` or `workflow_key` is required"
            task = Task(
                workflow=workflow_class,
                repeat_times=self.meta.repeat_times,
                format_args=copy.deepcopy(self.meta.format),
                rollout_args=copy.deepcopy(self.meta.rollout_args),
                workflow_args=copy.deepcopy(self.meta.workflow_args),
                reward_fn_args=copy.deepcopy(self.meta.reward_fn_args),
                is_eval=self.meta.task_type == TaskType.EVAL,
                reward_fn=reward_fn,
                raw_task=sample,
            )
            tasks.append(task)
        return tasks


@FILE_READERS.register_module("raw")
class RawDataReader(BaseFileReader):
    def __init__(self, meta: StorageConfig, config: Optional[BufferConfig]):
        self.returned = False
        self.dataset = load_dataset(meta.path, name=meta.subset_name, split=meta.split)

    def __len__(self):
        return len(self.dataset)

    def read(self, batch_size: Optional[int] = None) -> List:
        if self.returned:
            raise StopIteration
        self.returned = True
        return self.dataset.to_list()
