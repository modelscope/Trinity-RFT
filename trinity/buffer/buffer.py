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
        storage_config = self.buffer_mapping.get(dataset_name, None)
        if storage_config is None:
            raise ValueError(f"{dataset_name} not found.")
        return storage_config

    def register_dataset(self, storage_config: StorageConfig) -> None:
        if storage_config.name in self.buffer_mapping:
            raise ValueError(f"{storage_config.name} already exists.")
        self.buffer_mapping[storage_config.name] = storage_config


def get_buffer_reader(storage_config: StorageConfig, buffer_config: BufferConfig) -> BufferReader:
    """Get a buffer reader for the given dataset name."""
    if storage_config.storage_type == StorageType.SQL:
        from trinity.buffer.reader.sql_reader import SQLReader

        return SQLReader(storage_config, buffer_config)
    elif storage_config.storage_type == StorageType.QUEUE:
        from trinity.buffer.reader.queue_reader import QueueReader

        return QueueReader(storage_config, buffer_config)
    elif storage_config.storage_type == StorageType.FILE:
        from trinity.buffer.reader.file_reader import FILE_READERS

        file_read_type = storage_config.algorithm_type
        if file_read_type is not None:
            file_read_type = file_read_type.value
        else:
            file_read_type = "rollout"
        return FILE_READERS.get(file_read_type)(storage_config, buffer_config)
    else:
        raise ValueError(f"{storage_config.storage_type} not supported.")


def get_buffer_writer(storage_config: StorageConfig, buffer_config: BufferConfig) -> BufferWriter:
    """Get a buffer writer for the given dataset name."""
    if storage_config.storage_type == StorageType.SQL:
        from trinity.buffer.writer.sql_writer import SQLWriter

        return SQLWriter(storage_config, buffer_config)
    elif storage_config.storage_type == StorageType.QUEUE:
        from trinity.buffer.writer.queue_writer import QueueWriter

        return QueueWriter(storage_config, buffer_config)
    else:
        raise ValueError(f"{storage_config.storage_type} not supported.")
