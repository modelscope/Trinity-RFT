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
