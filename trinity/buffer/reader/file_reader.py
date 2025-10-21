"""Filed based buffer reader."""

from typing import Iterable, List, Optional, Union

import datasets
from datasets import Dataset, IterableDataset, load_dataset

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.schema.formatter import FORMATTER
from trinity.common.config import BufferConfig, StorageConfig


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
        shuffle: bool = False,
        base_seed: Optional[int] = 42,
    ):
        self.dataset_size = len(dataset)
        self.name = name
        self.current_batch_size = None
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.base_seed = base_seed

        self.current_offset = offset
        if self.shuffle:
            assert not isinstance(
                dataset, IterableDataset
            ), "Shuffle is not supported for IterableDataset"
            self.dataset = dataset.shuffle(seed=self.current_seed)
        else:
            self.dataset = dataset

        # convert epochs/steps to sample number
        if total_steps:
            self.total_samples = default_batch_size * total_steps
        else:
            if drop_last:
                self.num_per_epoch = self.dataset_size - (self.dataset_size % default_batch_size)
            else:
                self.num_per_epoch = self.dataset_size
            self.total_samples = self.num_per_epoch * total_epochs

        if enable_progress_bar:
            from ray.experimental.tqdm_ray import tqdm

            self.progress_bar = tqdm(
                total=self.total_samples,
                desc=f"Dataset [{self.name}] Progressing",
            )
        else:
            self.progress_bar = DummyProgressBar()

        self.progress_bar.update(self.current_offset)

    @property
    def current_seed(self):
        return self.base_seed + self.current_offset // self.dataset_size

    def read_batch(self, batch_size: int) -> Union[List, Iterable]:
        if self.current_offset >= self.total_samples:
            self.progress_bar.close()
            raise StopIteration
        start_epoch = self.current_offset // self.num_per_epoch
        start_index = self.current_offset % self.num_per_epoch

        batch = []
        for i in range(start_index, start_index + batch_size):
            if i < self.num_per_epoch:
                batch.append(self.dataset[i])
            else:
                assert not self.drop_last
                break

        self.current_offset += len(batch)
        self.progress_bar.update(len(batch))
        if start_epoch != self.current_offset // self.num_per_epoch:
            assert self.current_offset % self.num_per_epoch == 0
            if self.shuffle:
                self.dataset = self.dataset.shuffle(seed=self.current_seed)

        return batch, range(start_index, self.current_offset)

    def select_batch(self, indices: List[int]) -> List:
        batch = []
        for i in indices:
            assert 0 <= i < self.dataset_size
            batch.append(self.dataset[int(i)])
        return batch


class BaseFileReader(BufferReader):
    def __len__(self):
        return len(self.dataset.dataset)

    @property
    def index(self) -> int:
        return self.dataset.current_offset

    async def read_async(self, batch_size: Optional[int] = None):
        try:
            return self.read(batch_size)
        except StopIteration as e:
            raise StopAsyncIteration from e


class ExperienceFileReader(BaseFileReader):
    """Reader for SFT / DPO file data."""

    def __init__(self, meta: StorageConfig, config: BufferConfig):
        self.formatter = FORMATTER.get(meta.schema_type)(
            tokenizer_path=config.tokenizer_path, format_config=meta.format
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
        )

    def read(self, batch_size: Optional[int] = None) -> List:
        samples, _ = self.dataset.read_batch(batch_size or self.read_batch_size)
        exp_list = []
        for sample in samples:
            experience = self.formatter.format(sample)
            exp_list.append(experience)
        return exp_list


class TaskFileReader(BaseFileReader):
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
            total_epochs=self.meta.total_epochs if not self.meta.is_eval else 1,
            offset=self.meta.index,
            drop_last=not self.meta.is_eval,
            total_steps=meta.total_steps,
            enable_progress_bar=meta.enable_progress_bar,
            shuffle=meta.shuffle,
            base_seed=meta.seed,
        )
        self.formatter = FORMATTER.get("task")(meta)

    def read(self, batch_size: Optional[int] = None) -> List:
        batch_size = batch_size or self.read_batch_size
        tasks = []
        samples, indices = self.dataset.read_batch(batch_size)
        for sample, index in zip(samples, indices):
            task = self.formatter.format(sample)
            task.index.index = index
            tasks.append(task)
        return tasks

    def read_with_indices(self, indices: List[int]) -> List:
        """Read tasks with indices."""
        samples = self.dataset.select_batch(indices)
        tasks = []
        for index, sample in zip(indices, samples):
            task = self.formatter.format(sample)
            task.index.index = index
            tasks.append(task)
        return tasks

    async def read_with_indices_async(self, indices: List[int]) -> List:
        """Read tasks with indices asynchronously."""
        return self.read_with_indices(indices)
