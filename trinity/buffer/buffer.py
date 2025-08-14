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
        from trinity.buffer.reader.file_reader import FILE_READERS

        algorithm_type = storage_config.algorithm_type
        if storage_config.raw:
            file_read_type = "raw"
        elif algorithm_type is not None:
            file_read_type = algorithm_type
        else:
            file_read_type = "rollout"
        return FILE_READERS.get(file_read_type)(storage_config, buffer_config)  # type: ignore
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
