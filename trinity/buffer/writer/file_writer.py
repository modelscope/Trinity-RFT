"""Writer of the File buffer."""
import os
from typing import List

import jsonlines as jl

from trinity.buffer.buffer_writer import BufferWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class RawFileWriter(BufferWriter):
    """Writer of the Queue buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig):
        assert meta.storage_type == StorageType.FILE
        if meta.path is None:
            raise ValueError("File path cannot be None for RawFileWriter")
        ext = os.path.splitext(meta.path)[-1]
        if ext != ".jsonl":
            raise ValueError(f"File path must end with .json or .jsonl, got {meta.path}")
        self.writer = jl.open(meta.path, mode="a")

    def write(self, data: List) -> None:
        self.writer.write_all(data)

    def finish(self):
        self.writer.close()
