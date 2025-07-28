from abc import ABC, abstractmethod
from typing import Dict, List, Literal

import numpy as np

from trinity.buffer import BufferWriter
from trinity.common.experience import Experience
from trinity.utils.registry import Registry

ADD_STRATEGY = Registry("add_strategy")


class AddStrategy(ABC):
    def __init__(self, writer: BufferWriter, **kwargs) -> None:
        self.writer = writer

    @abstractmethod
    async def add(self, experiences: List[Experience], step: int) -> int:
        """Add experiences to the buffer.

        Args:
            experiences (`Experience`): The experiences to be added.
            step (`int`): The current step number.

        Returns:
            `int`: The number of experiences added to the buffer.
        """

    @classmethod
    @abstractmethod
    def default_args(cls) -> dict:
        """Get the default arguments of the add strategy.

        Returns:
            `dict`: The default arguments.
        """


@ADD_STRATEGY.register_module("reward_variance")
class RewardVarianceAddStrategy(AddStrategy):
    """An example AddStrategy that filters experiences based on a reward variance threshold."""

    def __init__(self, writer: BufferWriter, variance_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(writer)
        self.variance_threshold = variance_threshold

    async def add(self, experiences: List[Experience], step: int) -> int:
        cnt = 0
        grouped_experiences = group_by(experiences, id_type="task")
        for _, group_exps in grouped_experiences.items():
            if len(group_exps) < 2:
                continue
            # check if the rewards are the same
            rewards = [exp.reward for exp in group_exps]
            variance = np.var(rewards)
            if variance <= self.variance_threshold:
                continue
            cnt += len(group_exps)
            await self.writer.write_async(group_exps)
        return cnt

    @classmethod
    def default_args(cls) -> dict:
        return {"variance_threshold": 0.0}


@ADD_STRATEGY.register_module("duplicate_informative")
class DuplicateInformativeAddStrategy(AddStrategy):
    """An example AddStrategy that only adds experiences with non-zero advantage and repeats them to reach the target size
    Ref: POLARIS
    """

    def __init__(self, writer: BufferWriter, variance_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(writer)
        self.variance_threshold = variance_threshold

    async def add(self, experiences: List[Experience], step: int) -> int:
        cnt = 0
        cnt_tot = len(experiences)
        effective_tasks, effective_experiences = [], []
        grouped_experiences = group_by(experiences, id_type="task")
        for task_id, group_exps in grouped_experiences.items():
            if len(group_exps) < 2:
                continue
            # check if the rewards are the same
            rewards = [exp.reward for exp in group_exps]
            variance = np.var(rewards)
            if variance <= self.variance_threshold:
                continue
            cnt += len(group_exps)
            effective_tasks.append(task_id)
            effective_experiences.extend(group_exps)

        if not effective_tasks:
            return 0

        import copy
        import random

        task_ids_to_add = effective_tasks.copy()
        task_id_offset = len(grouped_experiences)
        while cnt < cnt_tot:
            if not task_ids_to_add:
                task_ids_to_add = effective_tasks.copy()
                task_id_offset += len(grouped_experiences)
            task_id = random.choice(task_ids_to_add)
            task_ids_to_add.remove(task_id)

            copied_exps = copy.deepcopy(grouped_experiences[task_id])

            for exp in copied_exps:
                exp.eid.task += task_id_offset

            cnt += len(copied_exps)
            effective_experiences.extend(copied_exps)

        await self.writer.write_async(effective_experiences)
        return cnt

    @classmethod
    def default_args(cls) -> dict:
        return {"variance_threshold": 0.0}


def group_by(
    experiences: List[Experience], id_type: Literal["task", "run", "step"]
) -> Dict[str, List[Experience]]:
    """Group experiences by ID."""
    if id_type == "task":
        id_type = "tid"
    elif id_type == "run":
        id_type = "rid"
    elif id_type == "step":
        id_type = "sid"
    else:
        raise ValueError(f"Unknown id_type: {id_type}")
    grouped = {}
    for exp in experiences:
        group_id = getattr(exp.eid, id_type)
        if group_id not in grouped:
            grouped[group_id] = []
        grouped[group_id].append(exp)
    return grouped
