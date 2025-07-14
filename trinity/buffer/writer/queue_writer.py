"""Writer of the Queue buffer."""
from typing import List

import ray

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.queue import QueueActor
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class QueueWriter(BufferWriter):
    """Writer of the Queue buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig):
        assert meta.storage_type == StorageType.QUEUE
        self.config = config
        self.queue = QueueActor.get_actor(meta, config)

    def write(self, data: List) -> None:
        ray.get(self.queue.put_batch.remote(data))

    async def write_async(self, data):
        return await self.queue.put_batch.remote(data)

    async def acquire(self) -> int:
        return await self.queue.acquire.remote()

    async def release(self) -> int:
        return await self.queue.release.remote()
