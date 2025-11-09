"""Filed based buffer reader."""

from typing import List, Optional

import datasets
from datasets import Dataset, load_dataset

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
        samples = self.dataset.read_batch(batch_size or self.read_batch_size)
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
        )
        self.formatter = FORMATTER.get("task")(meta)

    def read(self, batch_size: Optional[int] = None) -> List:
        batch_size = batch_size or self.read_batch_size
        tasks = []
        samples = self.dataset.read_batch(batch_size)
        for sample in samples:
            task = self.formatter.format(sample)
            tasks.append(task)
        return tasks


import os
def read_astune_config(yaml_fp):
    from hydra import initialize, compose
    from omegaconf import DictConfig

    def load_hydra_config(config_path: str, config_name: str) -> DictConfig:
        with initialize(config_path=config_path, version_base=None):
            cfg = compose(config_name=config_name, overrides=[])
            return cfg

    dir_path = os.path.dirname(yaml_fp)
    file_name = os.path.basename(yaml_fp)
    return load_hydra_config(config_path=dir_path, config_name=file_name)

class AstuneTaskReader(BaseFileReader):
    def __init__(self, meta: StorageConfig, config: BufferConfig):
        self.meta = meta
        self.read_batch_size = config.batch_size
        self.split = meta.split

        yaml_path = os.environ.get('ASTUNE_CONFIG_REDIRECT', None)
        if yaml_path is None:
            raise ValueError("ASTUNE_CONFIG_REDIRECT is not set in environment variables")
        astune_config = read_astune_config(os.path.relpath(yaml_path, os.path.dirname(__file__)))

        # from vsdb import bp
        # bp("XXX")

        from astune.task_reader.task_reader_base import TaskReaderRouter, task_to_standard_dataset
        task_reader = TaskReaderRouter(astune_config)
        if 'train' in self.split:
            train_dataset = task_to_standard_dataset(task_reader.get_training_tasks())
        if 'val' in self.split:
            train_dataset = task_to_standard_dataset(task_reader.get_validation_tasks())

        self.dataset = _HFBatchReader(
            datasets.concatenate_datasets([train_dataset]),
            name=meta.name,
            default_batch_size=self.read_batch_size,
            total_epochs=self.meta.total_epochs if not self.meta.is_eval else 1,
            offset=self.meta.index,
            drop_last=not self.meta.is_eval,
            total_steps=meta.total_steps,
            enable_progress_bar=meta.enable_progress_bar,
        )
        self.formatter = FORMATTER.get("task")(meta)

    def read(self, batch_size: Optional[int] = None) -> List:
        batch_size = batch_size or self.read_batch_size
        tasks = []
        samples = self.dataset.read_batch(batch_size)
        for sample in samples:
            task = self.formatter.format(sample)
            tasks.append(task)
        return tasks
