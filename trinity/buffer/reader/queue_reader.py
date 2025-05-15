"""Reader of the Queue buffer."""

from typing import List, Optional

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.queue import QueueActor
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import ReadStrategy, StorageType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class QueueReader(BufferReader):
    """Reader of the Queue buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig):
        assert meta.storage_type == StorageType.QUEUE
        self.config = config
        self.queue = QueueActor.options(
            name=f"queue-{meta.name}",
            get_if_exists=True,
        ).remote(meta, config)

    def read(self, strategy: Optional[ReadStrategy] = None) -> List:
        if strategy is not None and strategy != ReadStrategy.FIFO:
            raise NotImplementedError(f"Read strategy {strategy} not supported for Queue Reader.")
        try:
            exps = ray.get(self.queue.get_batch.remote(self.config.read_batch_size))
        except StopAsyncIteration:
            raise StopIteration()
        return exps
