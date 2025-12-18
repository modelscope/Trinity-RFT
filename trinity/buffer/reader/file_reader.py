"""Filed based buffer reader."""

from typing import List, Optional, Tuple

import datasets
from datasets import Dataset, load_dataset

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.schema import FORMATTER
from trinity.common.config import StorageConfig


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

    def read_batch(self, batch_size: int) -> Tuple[List, List]:
        batch, indices = [], []
        while len(batch) < batch_size:
            if self.current_offset >= self.total_samples:
                if not self.drop_last and len(batch) > 0:
                    break
                self.progress_bar.close()
                raise StopIteration
            index = self.current_offset % self.dataset_size
            batch.append(self.dataset[index])
            indices.append(index)
            self.current_offset += 1

        self.progress_bar.update(len(batch))
        return batch, indices

    def select_batch(self, indices: List[int]) -> List:
        batch = []
        for i in indices:
            assert 0 <= i < self.dataset_size
            batch.append(self.dataset[int(i)])
        self.progress_bar.update(len(batch))  # update progress bar
        return batch


class BaseFileReader(BufferReader):
    async def read_async(self, batch_size: Optional[int] = None, **kwargs):
        try:
            return self.read(batch_size)
        except StopIteration as e:
            raise StopAsyncIteration from e


class FileReader(BaseFileReader):
    """Provide a unified interface for Experience and Task file readers."""

    def __init__(self, config: StorageConfig):
        if config.schema_type and config.schema_type != "task":
            self.reader = ExperienceFileReader(config)
        else:
            self.reader = TaskFileReader(config)

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List:
        return self.reader.read(batch_size)

    def read_with_indices(self, indices: List[int]) -> List:
        """Read tasks with indices."""
        return self.reader.read_with_indices(indices)

    async def read_with_indices_async(self, indices: List[int]) -> List:
        """Read tasks with indices asynchronously."""
        return await self.reader.read_with_indices_async(indices)

    def state_dict(self):
        return self.reader.state_dict()

    def load_state_dict(self, state_dict):
        return self.reader.load_state_dict(state_dict)

    def __len__(self):
        return self.reader.__len__()


class ExperienceFileReader(BaseFileReader):
    """Reader for SFT / DPO file data."""

    def __init__(self, config: StorageConfig):
        self.formatter = FORMATTER.get(config.schema_type)(
            tokenizer_path=config.tokenizer_path, format_config=config.format
        )
        self.read_batch_size = config.batch_size
        self.dataset = _HFBatchReader(
            load_dataset(config.path, name=config.subset_name, split=config.split),
            name=config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=config.total_epochs,
            drop_last=True,
            total_steps=config.total_steps,
            enable_progress_bar=config.enable_progress_bar,
        )

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List:
        samples, _ = self.dataset.read_batch(batch_size or self.read_batch_size)
        exp_list = []
        for sample in samples:
            experience = self.formatter.format(sample)
            exp_list.append(experience)
        return exp_list

    def state_dict(self):
        return {"current_index": self.dataset.current_offset}

    def load_state_dict(self, state_dict):
        self.dataset.current_offset = state_dict["current_index"]

    def __len__(self):
        return self.dataset.dataset_size


class TaskFileReader(BaseFileReader):
    """A Reader for task file data."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.name = config.name
        self.epoch = 0
        datasets.disable_caching()
        self.read_batch_size = config.batch_size
        self.dataset = _HFBatchReader(
            load_dataset(self.config.path, name=self.config.subset_name, split=self.config.split),
            name=self.config.name,
            default_batch_size=self.read_batch_size,
            total_epochs=self.config.total_epochs if not self.config.is_eval else 1,
            offset=self.config.index,
            drop_last=not self.config.is_eval,
            total_steps=self.config.total_steps if not self.config.is_eval else None,
            enable_progress_bar=self.config.enable_progress_bar,
        )
        self.formatter = FORMATTER.get("task")(config)

    def _get_tasks(self, samples: List, indices: List) -> List:
        tasks = []
        for sample, index in zip(samples, indices):
            task = self.formatter.format(sample)
            task.index["index"] = int(index)
            tasks.append(task)
        return tasks

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List:
        batch_size = batch_size or self.read_batch_size
        samples, indices = self.dataset.read_batch(batch_size)
        return self._get_tasks(samples, indices)

    def read_with_indices(self, indices: List[int]) -> List:
        """Read tasks with indices."""
        samples = self.dataset.select_batch(indices)
        return self._get_tasks(samples, indices)

    async def read_with_indices_async(self, indices: List[int]) -> List:
        """Read tasks with indices asynchronously."""
        return self.read_with_indices(indices)

    def state_dict(self):
        return {"current_index": self.dataset.current_offset}

    def load_state_dict(self, state_dict):
        self.dataset.current_offset = state_dict["current_index"]

    def __len__(self):
        return self.dataset.dataset_size
