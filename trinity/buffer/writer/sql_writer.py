"""Writer of the SQL buffer."""

import ray

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.storage.sql import SQLStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class SQLWriter(BufferWriter):
    """Writer of the SQL buffer."""

    def __init__(self, config: StorageConfig) -> None:
        assert config.storage_type == StorageType.SQL
        # we only support write RFT algorithm buffer for now
        self.wrap_in_ray = config.wrap_in_ray
        self.db_wrapper = SQLStorage.get_wrapper(config)

    def write(self, data: list) -> None:
        if self.wrap_in_ray:
            ray.get(self.db_wrapper.write.remote(data))
        else:
            self.db_wrapper.write(data)

    async def write_async(self, data):
        if self.wrap_in_ray:
            ray.get(self.db_wrapper.write.remote(data))
        else:
            self.db_wrapper.write(data)

    async def acquire(self) -> int:
        if self.wrap_in_ray:
            return ray.get(self.db_wrapper.acquire.remote())
        else:
            return 0

    async def release(self) -> int:
        if self.wrap_in_ray:
            return ray.get(self.db_wrapper.release.remote())
        else:
            self.db_wrapper.release()
            return 0
