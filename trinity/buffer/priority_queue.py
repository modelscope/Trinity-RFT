"""An Async PriorityQueue."""
import asyncio
from collections import deque
from typing import List, Union

import numpy as np
from sortedcontainers import SortedDict

from trinity.common.experience import Experience
from trinity.utils.registry import Registry

PRIORITY_FUNC = Registry("priority_fn")


@PRIORITY_FUNC.register_module("linear_decay")
def linear_decay_priority(item: List[Experience], decay: float = 0.1):
    return item[0].info["model_version"] - decay * item[0].info["use_count"]  # type: ignore


class AsyncPriorityQueue:
    """
    An asynchronous priority queue that manages a fixed-size buffer of experience items.
    Items are prioritized using a user-defined function and reinserted after a cooldown period.

    Attributes:
        capacity (int): Maximum number of items the queue can hold.
        priority_groups (SortedDict): Maps priorities to deques of items with the same priority.
        priority_fn (callable): Function used to determine the priority of an item.
        reuse_cooldown_time (float): Delay before reusing an item (set to infinity to disable).
    """

    def __init__(
        self,
        capacity: int,
        reuse_cooldown_time: float = np.inf,
        priority_fn: str = "linear_decay",
        **kwargs,
    ):
        """
        Initialize the async priority queue.

        Args:
            capacity (int): The maximum number of items the queue can store.
            reuse_cooldown_time (float): Time to wait before reusing an item.
        """
        self.capacity = capacity
        self._count = 0
        self.priority_groups = SortedDict()  # Maps priority -> deque of items
        priority_fn = PRIORITY_FUNC.get(priority_fn)
        from trinity.buffer.queue import QueueActor

        self.FINISH_MESSAGE = QueueActor.FINISH_MESSAGE

        self.priority_fn = (
            lambda item: priority_fn(item, **kwargs) if item != self.FINISH_MESSAGE else -np.inf  # type: ignore
        )
        self.reuse_cooldown_time = reuse_cooldown_time
        self._condition = asyncio.Condition()  # For thread-safe operations

    async def put(self, item: Union[List[Experience], str], delay: float = 0.0) -> None:
        """
        Insert an item into the queue, possibly replacing the lowest-priority item if full.

        Args:
            item (List[Experience]): A list of experiences to add.
            delay (float): Optional delay before insertion (for simulating timing behavior).
        """
        if delay > 0:
            await asyncio.sleep(delay)

        priority = self.priority_fn(item)
        async with self._condition:
            if self._count == self.capacity:
                # If full, only insert if new item has higher or equal priority than the lowest
                lowest_priority, item_queue = self.priority_groups.peekitem(index=0)
                if lowest_priority > priority:
                    return  # Skip insertion if lower priority
                # Remove the lowest priority item
                item_queue.popleft()
                if not item_queue:
                    self.priority_groups.popitem(index=0)
                self._count -= 1

            # Add the new item
            if priority not in self.priority_groups:
                self.priority_groups[priority] = deque()
            self.priority_groups[priority].append(item)
            self._count += 1
            self._condition.notify_all()

    async def get(self) -> List[Experience]:
        """
        Retrieve the highest-priority item from the queue.

        Returns:
            List[Experience]: The highest-priority item (list of experiences).

        Notes:
            - After retrieval, the item is optionally reinserted after a cooldown period.
        """
        async with self._condition:
            while self._count == 0:
                await self._condition.wait_for(lambda: self._count > 0)

            _, item_queue = self.priority_groups.peekitem(index=-1)
            item = item_queue.popleft()

            if not item_queue:
                self.priority_groups.popitem(index=-1)

            self._count -= 1

        if item != self.FINISH_MESSAGE:
            for exp in item:
                exp.info["use_count"] += 1
        # Optionally resubmit the item after a cooldown
        if not np.isinf(self.reuse_cooldown_time):
            asyncio.create_task(self.put(item, self.reuse_cooldown_time))

        return item

    def size(self) -> int:
        """
        Get the current number of items in the queue.

        Returns:
            int: Number of items currently stored.
        """
        return self._count
