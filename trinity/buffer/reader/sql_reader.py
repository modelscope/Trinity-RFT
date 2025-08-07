"""Reader of the SQL buffer."""

from typing import List, Optional

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.ray_wrapper import DBWrapper
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import ReadStrategy, StorageType


class SQLReader(BufferReader):
    """Reader of the SQL buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig) -> None:
        assert meta.storage_type == StorageType.SQL
        self.wrap_in_ray = meta.wrap_in_ray
        self.db_wrapper = DBWrapper.get_wrapper(meta, config)

    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        if self.wrap_in_ray:
            return ray.get(self.db_wrapper.read.remote(batch_size, strategy))
        else:
            return self.db_wrapper.read(batch_size, strategy)

    async def read_async(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        if self.wrap_in_ray:
            return await self.db_wrapper.read.remote(batch_size, strategy)
        else:
            return self.db_wrapper.read(batch_size, strategy)
