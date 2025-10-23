"""Reader of the Queue buffer."""

from typing import List, Optional

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.storage.queue import QueueStorage
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType


class QueueReader(BufferReader):
    """Reader of the Queue buffer."""

    def __init__(self, storage_config: StorageConfig, config: BufferConfig):
        assert storage_config.storage_type == StorageType.QUEUE
        self.timeout = storage_config.max_read_timeout
        self.read_batch_size = config.train_batch_size
        self.queue = QueueStorage.get_wrapper(storage_config, config)

    def read(self, batch_size: Optional[int] = None) -> List:
        try:
            batch_size = batch_size or self.read_batch_size
            exps = ray.get(self.queue.get_batch.remote(batch_size, timeout=self.timeout))
            if len(exps) != batch_size:
                raise TimeoutError(
                    f"Read incomplete batch ({len(exps)}/{batch_size}), please check your workflow."
                )
        except StopAsyncIteration:
            raise StopIteration()
        return exps

    async def read_async(self, batch_size: Optional[int] = None) -> List:
        batch_size = batch_size or self.read_batch_size
        exps = await self.queue.get_batch.remote(batch_size, timeout=self.timeout)
        if len(exps) != batch_size:
            raise TimeoutError(
                f"Read incomplete batch ({len(exps)}/{batch_size}), please check your workflow."
            )
        return exps
