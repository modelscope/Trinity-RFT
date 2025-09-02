# -*- coding: utf-8 -*-
"""The buffer module"""
from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.buffer_writer import BufferWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType


def get_buffer_reader(storage_config: StorageConfig, buffer_config: BufferConfig) -> BufferReader:
    """Get a buffer reader for the given dataset name."""
    if storage_config.storage_type == StorageType.SQL:
        from trinity.buffer.reader.sql_reader import SQLReader

        return SQLReader(storage_config, buffer_config)
    elif storage_config.storage_type == StorageType.QUEUE:
        from trinity.buffer.reader.queue_reader import QueueReader

        return QueueReader(storage_config, buffer_config)
    elif storage_config.storage_type == StorageType.FILE:
        from trinity.buffer.reader.file_reader import (
            ExperienceFileReader,
            TaskFileReader,
        )

        schema_type = storage_config.schema_type
        if schema_type:
            # only trainer input has schema type
            return ExperienceFileReader(storage_config, buffer_config)
        else:
            return TaskFileReader(storage_config, buffer_config)
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
    elif storage_config.storage_type == StorageType.FILE:
        from trinity.buffer.writer.file_writer import JSONWriter

        return JSONWriter(storage_config, buffer_config)
    else:
        raise ValueError(f"{storage_config.storage_type} not supported.")
