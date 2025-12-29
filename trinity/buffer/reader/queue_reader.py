"""Reader of the Queue buffer."""

from typing import Dict, List, Optional

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.storage.queue import QueueStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class QueueReader(BufferReader):
    """Reader of the Queue buffer."""

    def __init__(self, config: StorageConfig):
        assert config.storage_type == StorageType.QUEUE.value
        self.timeout = config.max_read_timeout
        self.read_batch_size = config.batch_size
        self.queue = QueueStorage.get_wrapper(config)

    def read(self, batch_size: Optional[int] = None, **kwargs) -> List:
        try:
            batch_size = batch_size or self.read_batch_size
            exps = ray.get(self.queue.get_batch.remote(batch_size, timeout=self.timeout, **kwargs))
            if len(exps) != batch_size:
                raise TimeoutError(
                    f"Read incomplete batch ({len(exps)}/{batch_size}), please check your workflow."
                )
        except StopAsyncIteration:
            raise StopIteration()
        return exps

    async def read_async(self, batch_size: Optional[int] = None, **kwargs) -> List:
        batch_size = batch_size or self.read_batch_size
        exps = await self.queue.get_batch.remote(batch_size, timeout=self.timeout, **kwargs)
        if len(exps) != batch_size:
            raise TimeoutError(
                f"Read incomplete batch ({len(exps)}/{batch_size}), please check your workflow."
            )
        return exps

    def state_dict(self) -> Dict:
        # Queue Not supporting state dict yet
        return {"current_index": 0}

    def load_state_dict(self, state_dict):
        # Queue Not supporting state dict yet
        return None
