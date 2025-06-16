"""A queue implemented by Ray Actor."""
import asyncio
from copy import deepcopy
from typing import List

import ray

from trinity.buffer.writer.file_writer import JSONWriter
from trinity.buffer.writer.sql_writer import SQLWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType


def is_database_url(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ["sqlite:///", "postgresql://", "mysql://"])


def is_json_file(path: str) -> bool:
    return path.endswith(".json") or path.endswith(".jsonl")


@ray.remote
class QueueActor:
    """An asyncio.Queue based queue actor."""

    FINISH_MESSAGE = "$FINISH$"

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        self.config = config
        self.capacity = getattr(config, "capacity", 10000)
        self.queue = asyncio.Queue(self.capacity)
        if storage_config.path is not None and len(storage_config.path) > 0:
            if is_database_url(storage_config.path):
                storage_config.storage_type = StorageType.SQL
                sql_config = deepcopy(storage_config)
                sql_config.storage_type = StorageType.SQL
                sql_config.wrap_in_ray = False
                self.writer = SQLWriter(sql_config, self.config)
            elif is_json_file(storage_config.path):
                storage_config.storage_type = StorageType.FILE
                json_config = deepcopy(storage_config)
                json_config.storage_type = StorageType.FILE
                self.writer = JSONWriter(json_config, self.config)
            else:
                self.writer = None
        else:
            self.writer = None

    def length(self) -> int:
        """The length of the queue."""
        return self.queue.qsize()

    async def put_batch(self, exp_list: List) -> None:
        """Put batch of experience."""
        await self.queue.put(exp_list)
        if self.writer is not None:
            self.writer.write(exp_list)

    async def finish(self) -> None:
        """Stop the queue."""
        await self.queue.put(self.FINISH_MESSAGE)

    async def get_batch(self, batch_size: int) -> List:
        """Get batch of experience."""
        batch = []
        while True:
            exp_list = await self.queue.get()
            if exp_list == self.FINISH_MESSAGE:
                raise StopAsyncIteration()
            batch.extend(exp_list)
            if len(batch) >= batch_size:
                break
        return batch
