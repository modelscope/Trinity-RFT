import json
import os
from typing import List

from trinity.buffer.buffer_writer import BufferWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.common.workflows import Task


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Experience):
            return o.to_dict()
        if isinstance(o, Task):
            return o.to_dict()
        return super().default(o)


class JSONWriter(BufferWriter):
    def __init__(self, meta: StorageConfig, config: BufferConfig):
        assert meta.storage_type == StorageType.FILE
        if meta.path is None:
            raise ValueError("File path cannot be None for RawFileWriter")
        ext = os.path.splitext(meta.path)[-1]
        if ext != ".jsonl" and ext != ".json":
            raise ValueError(f"File path must end with .json or .jsonl, got {meta.path}")
        path_dir = os.path.dirname(meta.path)
        os.makedirs(path_dir, exist_ok=True)
        self.file = open(meta.path, "a", encoding="utf-8")
        self.encoder = _Encoder(ensure_ascii=False)

    def write(self, data: List) -> None:
        for item in data:
            json_str = self.encoder.encode(item)
            self.file.write(json_str + "\n")
        self.file.flush()

    def finish(self):
        self.file.close()
