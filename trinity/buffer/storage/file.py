"""File Storage"""
import json
import os
from typing import List

import ray

from trinity.buffer.utils import default_storage_path
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.experience import EID, Experience
from trinity.common.workflows import Task


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Experience):
            return o.to_dict()
        if isinstance(o, Task):
            return o.to_dict()
        if isinstance(o, EID):
            return o.to_dict()
        return super().default(o)


class FileStorage:
    """
    A wrapper of a local jsonl file.

    If `wrap_in_ray` in `StorageConfig` is `True`, this class will be run as
    a Ray Actor, and provide a remote interface to the local file.

    This wrapper is only for writing, if you want to read from the file, use
    StorageType.QUEUE instead.
    """

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        if storage_config.path is None:
            storage_config.path = default_storage_path(storage_config, config)
        ext = os.path.splitext(storage_config.path)[-1]
        if ext != ".jsonl" and ext != ".json":
            raise ValueError(
                f"File path must end with '.json' or '.jsonl', got {storage_config.path}"
            )
        path_dir = os.path.dirname(os.path.abspath(storage_config.path))
        os.makedirs(path_dir, exist_ok=True)
        self.file = open(storage_config.path, "a", encoding="utf-8")
        self.encoder = _Encoder(ensure_ascii=False)
        self.ref_count = 0

    @classmethod
    def get_wrapper(cls, storage_config: StorageConfig, config: BufferConfig):
        if storage_config.wrap_in_ray:
            return (
                ray.remote(cls)
                .options(
                    name=f"json-{storage_config.name}",
                    namespace=storage_config.ray_namespace or ray.get_runtime_context().namespace,
                    get_if_exists=True,
                )
                .remote(storage_config, config)
            )
        else:
            return cls(storage_config, config)

    def write(self, data: List) -> None:
        for item in data:
            json_str = self.encoder.encode(item)
            self.file.write(json_str + "\n")
        self.file.flush()

    def read(self) -> List:
        raise NotImplementedError(
            "read() is not implemented for FILE Storage, please use QUEUE instead"
        )

    def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    def release(self) -> int:
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.file.close()
        return self.ref_count
