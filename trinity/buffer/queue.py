"""A queue implemented by Ray Actor."""
import asyncio
from copy import deepcopy
from typing import List

import ray

from trinity.buffer.priority_queue import AsyncPriorityQueue
from trinity.buffer.writer.file_writer import JSONWriter
from trinity.buffer.writer.sql_writer import SQLWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.utils.log import get_logger


def is_database_url(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ["sqlite:///", "postgresql://", "mysql://"])


def is_json_file(path: str) -> bool:
    return path.endswith(".json") or path.endswith(".jsonl")


class QueueActor:
    """An asyncio.Queue based queue actor."""

    FINISH_MESSAGE = "$FINISH$"

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        self.logger = get_logger(__name__)
        self.config = config
        self.capacity = storage_config.capacity
        if storage_config.use_priority_queue:
            reuse_cooldown_time = storage_config.reuse_cooldown_time
            replay_buffer_kwargs = storage_config.replay_buffer_kwargs
            self.queue = AsyncPriorityQueue(
                self.capacity, reuse_cooldown_time, **replay_buffer_kwargs
            )
        else:
            self.queue = asyncio.Queue(self.capacity)
        st_config = deepcopy(storage_config)
        st_config.wrap_in_ray = False
        if st_config.path is not None:
            if is_database_url(st_config.path):
                st_config.storage_type = StorageType.SQL
                self.writer = SQLWriter(st_config, self.config)
            elif is_json_file(st_config.path):
                st_config.storage_type = StorageType.FILE
                self.writer = JSONWriter(st_config, self.config)
            else:
                self.logger.warning("Unknown supported storage path: %s", st_config.path)
                self.writer = None
        else:
            st_config.storage_type = StorageType.FILE
            self.writer = JSONWriter(st_config, self.config)
        self.logger.warning(f"Save experiences in {st_config.path}.")
        self.ref_count = 0

    async def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    async def release(self) -> int:
        """Release the queue."""
        self.ref_count -= 1
        if self.ref_count <= 0:
            await self.queue.put(self.FINISH_MESSAGE)
            await self.writer.release()
        return self.ref_count

    def length(self) -> int:
        """The length of the queue."""
        return self.queue.qsize()

    async def put_batch(self, exp_list: List) -> None:
        """Put batch of experience."""
        await self.queue.put(exp_list)
        if self.writer is not None:
            self.writer.write(exp_list)

    async def get_batch(self, batch_size: int, timeout: float) -> List:
        """Get batch of experience."""
        batch = []
        while True:
            try:
                exp_list = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Timeout when waiting for experience, only get {len(batch)} experiences.\n"
                    "This phenomenon is usually caused by the workflow not returning enough "
                    "experiences or running timeout. Please check your workflow implementation."
                )
                return batch
            if exp_list == self.FINISH_MESSAGE:
                raise StopAsyncIteration()
            batch.extend(exp_list)
            if len(batch) >= batch_size:
                break
        return batch

    @classmethod
    def get_actor(cls, storage_config: StorageConfig, config: BufferConfig):
        """Get the queue actor."""
        return (
            ray.remote(cls)
            .options(
                name=f"queue-{storage_config.name}",
                namespace=storage_config.ray_namespace or ray.get_runtime_context().namespace,
                get_if_exists=True,
            )
            .remote(storage_config, config)
        )
