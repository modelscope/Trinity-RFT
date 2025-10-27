# -*- coding: utf-8 -*-
"""The buffer module"""
from typing import Union

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.buffer_writer import BufferWriter
from trinity.common.config import ExperienceBufferConfig, StorageConfig, TasksetConfig
from trinity.common.constants import StorageType

BufferStorageConfig = Union[TasksetConfig, ExperienceBufferConfig, StorageConfig]


def get_buffer_reader(config: BufferStorageConfig) -> BufferReader:
    """Get a buffer reader for the given dataset name."""
    if not isinstance(config, StorageConfig):
        storage_config: StorageConfig = config.to_storage_config()
    else:
        storage_config = config
    if storage_config.storage_type == StorageType.SQL:
        from trinity.buffer.reader.sql_reader import SQLReader

        return SQLReader(storage_config)
    elif storage_config.storage_type == StorageType.QUEUE:
        from trinity.buffer.reader.queue_reader import QueueReader

        return QueueReader(storage_config)
    elif storage_config.storage_type == StorageType.FILE:
        from trinity.buffer.reader.file_reader import (
            ExperienceFileReader,
            TaskFileReader,
        )

        schema_type = storage_config.schema_type
        if schema_type:
            # only trainer input has schema type
            return ExperienceFileReader(storage_config)
        else:
            return TaskFileReader(storage_config)
    else:
        raise ValueError(f"{storage_config.storage_type} not supported.")


def get_buffer_writer(config: BufferStorageConfig) -> BufferWriter:
    """Get a buffer writer for the given dataset name."""
    if not isinstance(config, StorageConfig):
        storage_config: StorageConfig = config.to_storage_config()
    else:
        storage_config = config
    if storage_config.storage_type == StorageType.SQL:
        from trinity.buffer.writer.sql_writer import SQLWriter

        return SQLWriter(storage_config)
    elif storage_config.storage_type == StorageType.QUEUE:
        from trinity.buffer.writer.queue_writer import QueueWriter

        return QueueWriter(storage_config)
    elif storage_config.storage_type == StorageType.FILE:
        from trinity.buffer.writer.file_writer import JSONWriter

        return JSONWriter(storage_config)
    else:
        raise ValueError(f"{storage_config.storage_type} not supported.")
