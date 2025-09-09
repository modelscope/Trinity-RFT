"""Ray Queue storage"""
import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from functools import partial
from typing import List, Optional

import ray
from sortedcontainers import SortedDict

from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry


def is_database_url(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ["sqlite:///", "postgresql://", "mysql://"])


def is_json_file(path: str) -> bool:
    return path.endswith(".json") or path.endswith(".jsonl")


PRIORITY_FUNC = Registry("priority_fn")


@PRIORITY_FUNC.register_module("linear_decay")
def linear_decay_priority(item: List[Experience], decay: float = 0.1):
    return item[0].info["model_version"] - decay * item[0].info["use_count"]  # type: ignore


class QueueBuffer(ABC):
    @abstractmethod
    async def put(self, exps: List[Experience]) -> None:
        """Put a list of experiences into the queue."""

    @abstractmethod
    async def get(self) -> List[Experience]:
        """Get a list of experience from the queue."""

    @abstractmethod
    def qsize(self) -> int:
        """Get the current size of the queue."""

    @abstractmethod
    async def close(self) -> None:
        """Close the queue."""

    @abstractmethod
    def stopped(self) -> bool:
        """Check if there is no more data to read."""

    @classmethod
    def get_queue(cls, storage_config: StorageConfig, config: BufferConfig) -> "QueueBuffer":
        """Get a queue instance based on the storage configuration."""
        logger = get_logger(__name__)
        if storage_config.use_priority_queue:
            reuse_cooldown_time = storage_config.reuse_cooldown_time
            replay_buffer_kwargs = storage_config.replay_buffer_kwargs
            capacity = min(storage_config.capacity, config.train_batch_size * 2)
            logger.info(
                f"Using AsyncPriorityQueue with capacity {capacity}, reuse_cooldown_time {reuse_cooldown_time}."
            )
            return AsyncPriorityQueue(capacity, reuse_cooldown_time, **replay_buffer_kwargs)
        else:
            return AsyncQueue(capacity=storage_config.capacity)


class AsyncQueue(asyncio.Queue, QueueBuffer):
    def __init__(self, capacity: int):
        """
        Initialize the async queue with a specified capacity.

        Args:
            capacity (`int`): The maximum number of items the queue can hold.
        """
        super().__init__(maxsize=capacity)
        self._closed = False

    async def close(self) -> None:
        """Close the queue."""
        self._closed = True
        for getter in self._getters:
            if not getter.done():
                getter.set_exception(StopAsyncIteration())
        self._getters.clear()

    def stopped(self) -> bool:
        """Check if there is no more data to read."""
        return self._closed and self.empty()


class AsyncPriorityQueue(QueueBuffer):
    """
    An asynchronous priority queue that manages a fixed-size buffer of experience items.
    Items are prioritized using a user-defined function and reinserted after a cooldown period.

    Attributes:
        capacity (int): Maximum number of items the queue can hold. This value is automatically
            adjusted to be at most twice the read batch size.
        priority_groups (SortedDict): Maps priorities to deques of items with the same priority.
        priority_fn (callable): Function used to determine the priority of an item.
        reuse_cooldown_time (float): Delay before reusing an item (set to infinity to disable).
    """

    def __init__(
        self,
        capacity: int,
        reuse_cooldown_time: Optional[float] = None,
        priority_fn: str = "linear_decay",
        **kwargs,
    ):
        """
        Initialize the async priority queue.

        Args:
            capacity (`int`): The maximum number of items the queue can store.
            reuse_cooldown_time (`float`): Time to wait before reusing an item. Set to None to disable reuse.
            priority_fn (`str`): Name of the function to use for determining item priority.
            kwargs: Additional keyword arguments for the priority function.
        """
        self.capacity = capacity
        self.priority_groups = SortedDict()  # Maps priority -> deque of items
        self.priority_fn = partial(PRIORITY_FUNC.get(priority_fn), **kwargs)
        self.reuse_cooldown_time = reuse_cooldown_time
        self._condition = asyncio.Condition()  # For thread-safe operations
        self._closed = False

    async def _put(self, item: List[Experience], delay: float = 0) -> None:
        """
        Insert an item into the queue, replacing the lowest-priority item if full.

        Args:
            item (`List[Experience]`): A list of experiences to add.
            delay (`float`): Optional delay before insertion (for simulating timing behavior).
        """
        if delay > 0:
            await asyncio.sleep(delay)
        if len(item) == 0:
            return
        priority = self.priority_fn(item=item)
        async with self._condition:
            if len(self.priority_groups) == self.capacity:
                # If full, only insert if new item has higher or equal priority than the lowest
                lowest_priority, item_queue = self.priority_groups.peekitem(index=0)
                if lowest_priority > priority:
                    return  # Skip insertion if lower priority
                # Remove the lowest priority item
                item_queue.popleft()
                if not item_queue:
                    self.priority_groups.popitem(index=0)

            # Add the new item
            if priority not in self.priority_groups:
                self.priority_groups[priority] = deque()
            self.priority_groups[priority].append(item)
            self._condition.notify()

    async def put(self, item: List[Experience]) -> None:
        await self._put(item, delay=0)

    async def get(self) -> List[Experience]:
        """
        Retrieve the highest-priority item from the queue.

        Returns:
            List[Experience]: The highest-priority item (list of experiences).

        Notes:
            - After retrieval, the item is optionally reinserted after a cooldown period.
        """
        async with self._condition:
            while len(self.priority_groups) == 0:
                if self._closed:
                    raise StopAsyncIteration()
                await self._condition.wait()

            _, item_queue = self.priority_groups.peekitem(index=-1)
            item = item_queue.popleft()
            if not item_queue:
                self.priority_groups.popitem(index=-1)

        for exp in item:
            exp.info["use_count"] += 1
        # Optionally resubmit the item after a cooldown
        if self.reuse_cooldown_time is not None:
            asyncio.create_task(self._put(item, self.reuse_cooldown_time))

        return item

    def qsize(self):
        return len(self.priority_groups)

    async def close(self) -> None:
        """
        Close the queue.
        """
        async with self._condition:
            self._closed = True
            # No more items will be added, but existing items can still be processed.
            self.reuse_cooldown_time = None
            self._condition.notify_all()

    def stopped(self) -> bool:
        return self._closed and len(self.priority_groups) == 0


class QueueStorage:
    """An wrapper of a async queue."""

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        self.logger = get_logger(f"queue_{storage_config.name}", in_ray_actor=True)
        self.config = config
        self.capacity = storage_config.capacity
        self.queue = QueueBuffer.get_queue(storage_config, config)
        st_config = deepcopy(storage_config)
        st_config.wrap_in_ray = False
        if st_config.path is not None:
            if is_database_url(st_config.path):
                from trinity.buffer.writer.sql_writer import SQLWriter

                st_config.storage_type = StorageType.SQL
                self.writer = SQLWriter(st_config, self.config)
            elif is_json_file(st_config.path):
                from trinity.buffer.writer.file_writer import JSONWriter

                st_config.storage_type = StorageType.FILE
                self.writer = JSONWriter(st_config, self.config)
            else:
                self.logger.warning("Unknown supported storage path: %s", st_config.path)
                self.writer = None
        else:
            from trinity.buffer.writer.file_writer import JSONWriter

            st_config.storage_type = StorageType.FILE
            self.writer = JSONWriter(st_config, self.config)
        self.logger.warning(f"Save experiences in {st_config.path}.")
        self.ref_count = 0
        self.exp_pool = deque()  # A pool to store experiences
        self.closed = False

    async def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    async def release(self) -> int:
        """Release the queue."""
        self.ref_count -= 1
        if self.ref_count <= 0:
            await self.queue.close()
            if self.writer is not None:
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
        start_time = time.time()
        while len(self.exp_pool) < batch_size:
            if self.queue.stopped():
                # If the queue is stopped, ignore the rest of the experiences in the pool
                raise StopAsyncIteration("Queue is closed and no more items to get.")
            try:
                exp_list = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.exp_pool.extend(exp_list)
            except asyncio.TimeoutError:
                if time.time() - start_time > timeout:
                    self.logger.error(
                        f"Timeout when waiting for experience, only get {len(self.exp_pool)} experiences.\n"
                        "This phenomenon is usually caused by the workflow not returning enough "
                        "experiences or running timeout. Please check your workflow implementation."
                    )
                    batch = list(self.exp_pool)
                    self.exp_pool.clear()
                    return batch
        return [self.exp_pool.popleft() for _ in range(batch_size)]

    @classmethod
    def get_wrapper(cls, storage_config: StorageConfig, config: BufferConfig):
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
