# -*- coding: utf-8 -*-
"""The buffer module"""
import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.buffer_writer import BufferWriter
from trinity.common.config import BufferConfig, Config, StorageConfig
from trinity.common.constants import StorageType


@ray.remote(name="buffer")
class Buffer:
    """Responsible for storing experiences."""

    def __init__(self, config: Config):
        self.buffer_mapping: dict[str, StorageConfig] = {}
        self._register_from_config(config)

    def get_dataset_info(self, dataset_name: str) -> StorageConfig:
        dataset_config = self.buffer_mapping.get(dataset_name, None)
        if dataset_config is None:
            raise ValueError(f"{dataset_name} not found.")
        return dataset_config

    def register_dataset(self, dataset_config: StorageConfig) -> None:
        if dataset_config.name in self.buffer_mapping:
            raise ValueError(f"{dataset_config.name} already exists.")
        self.buffer_mapping[dataset_config.name] = dataset_config


def get_buffer_reader(dataset_config: StorageConfig, buffer_config: BufferConfig) -> BufferReader:
    """Get a buffer reader for the given dataset name."""
    if dataset_config.storage_type == StorageType.SQL:
        from trinity.buffer.reader.sql_reader import SQLReader

        return SQLReader(dataset_config, buffer_config)
    elif dataset_config.storage_type == StorageType.QUEUE:
        from trinity.buffer.reader.queue_reader import QueueReader

        return QueueReader(dataset_config, buffer_config)
    elif dataset_config.storage_type == StorageType.FILE:
        from trinity.buffer.reader.file_reader import FileReaderManager

        return FileReaderManager.create_reader(dataset_config, buffer_config)
    else:
        raise ValueError(f"{dataset_config.storage_type} not supported.")


def get_buffer_writer(dataset_config: StorageConfig, buffer_config: BufferConfig) -> BufferWriter:
    """Get a buffer writer for the given dataset name."""
    if dataset_config.storage_type == StorageType.SQL:
        from trinity.buffer.writer.sql_writer import SQLWriter

        return SQLWriter(dataset_config, buffer_config)
    elif dataset_config.storage_type == StorageType.QUEUE:
        from trinity.buffer.writer.queue_writer import QueueWriter

        return QueueWriter(dataset_config, buffer_config)
    else:
        raise ValueError(f"{dataset_config.storage_type} not supported.")
