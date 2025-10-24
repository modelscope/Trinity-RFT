from typing import List

import ray

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.storage.file import FileStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class JSONWriter(BufferWriter):
    def __init__(self, config: StorageConfig):
        assert config.storage_type == StorageType.FILE
        self.writer = FileStorage.get_wrapper(config)
        self.wrap_in_ray = config.wrap_in_ray

    def write(self, data: List) -> None:
        if self.wrap_in_ray:
            ray.get(self.writer.write.remote(data))
        else:
            self.writer.write(data)

    async def write_async(self, data):
        if self.wrap_in_ray:
            await self.writer.write.remote(data)
        else:
            self.writer.write(data)

    async def acquire(self) -> int:
        if self.wrap_in_ray:
            return await self.writer.acquire.remote()
        else:
            return 0

    async def release(self) -> int:
        if self.wrap_in_ray:
            return await self.writer.release.remote()
        else:
            self.writer.release()
            return 0
