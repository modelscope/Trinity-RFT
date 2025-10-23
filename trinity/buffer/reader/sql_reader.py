"""Reader of the SQL buffer."""

from typing import List, Optional

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.storage.sql import SQLStorage
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType


class SQLReader(BufferReader):
    """Reader of the SQL buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig) -> None:
        assert meta.storage_type == StorageType.SQL
        self.wrap_in_ray = meta.wrap_in_ray
        self.storage = SQLStorage.get_wrapper(meta, config)

    def read(self, batch_size: Optional[int] = None) -> List:
        if self.wrap_in_ray:
            return ray.get(self.storage.read.remote(batch_size))
        else:
            return self.storage.read(batch_size)

    async def read_async(self, batch_size: Optional[int] = None) -> List:
        if self.wrap_in_ray:
            try:
                return ray.get(self.storage.read.remote(batch_size))
            except StopIteration:
                raise StopAsyncIteration
        else:
            return self.storage.read(batch_size)
