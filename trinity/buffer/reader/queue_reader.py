"""Reader of the Queue buffer."""

from typing import List, Optional

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.ray_wrapper import QueueWrapper
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import ReadStrategy, StorageType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class QueueReader(BufferReader):
    """Reader of the Queue buffer."""

    def __init__(self, storage_config: StorageConfig, config: BufferConfig):
        assert storage_config.storage_type == StorageType.QUEUE
        self.timeout = storage_config.max_read_timeout
        self.read_batch_size = config.read_batch_size
        self.queue = QueueWrapper.get_wrapper(storage_config, config)

    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        if strategy is not None and strategy != ReadStrategy.FIFO:
            raise NotImplementedError(f"Read strategy {strategy} not supported for Queue Reader.")
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

    async def read_async(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        if strategy is not None and strategy != ReadStrategy.FIFO:
            raise NotImplementedError(f"Read strategy {strategy} not supported for Queue Reader.")
        try:
            batch_size = batch_size or self.read_batch_size
            exps = await self.queue.get_batch.remote(batch_size, timeout=self.timeout)
            if len(exps) != batch_size:
                raise TimeoutError(
                    f"Read incomplete batch ({len(exps)}/{batch_size}), please check your workflow."
                )
        except StopAsyncIteration:
            raise StopIteration()
        return exps
