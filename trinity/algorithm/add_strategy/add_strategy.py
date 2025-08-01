import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch

from trinity.buffer import BufferWriter
from trinity.common.experience import Experience
from trinity.utils.monitor import gather_metrics
from trinity.utils.registry import Registry
from trinity.utils.timer import Timer

ADD_STRATEGY = Registry("add_strategy")


class AddStrategy(ABC):
    def __init__(self, writer: BufferWriter, **kwargs) -> None:
        self.writer = writer

    @abstractmethod
    async def add(self, experiences: List[Experience], step: int) -> Tuple[int, Dict]:
        """Add experiences to the buffer.

        Args:
            experiences (`Experience`): The experiences to be added.
            step (`int`): The current step number.

        Returns:
            `int`: The number of experiences added to the buffer.
            `Dict`: Metrics for logging.
        """

    @classmethod
    @abstractmethod
    def default_args(cls) -> dict:
        """Get the default arguments of the add strategy.

        Returns:
            `dict`: The default arguments.
        """


class GroupAdvantageStrategy(AddStrategy):
    """An example AddStrategy that calculates group advantages."""

    @abstractmethod
    def group_experiences(self, exps: List[Experience]) -> Dict[str, List[Experience]]:
        """Group experiences by a certain criterion.

        Args:
            exps (List[Experience]): List of experiences to be grouped.

        Returns:
            Dict[str, List[Experience]]: A dictionary where keys are group identifiers and values are lists of experiences.
        """

    @abstractmethod
    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        """Calculate advantages for a group of experiences.

        Args:
            group_id (str): The identifier for the group of experiences.
            exps (List[Experience]): List of experiences in the group.

        Returns:
            Tuple[List[Experience], Dict]: A tuple containing the modified list of experiences and a dictionary of metrics.
        """

    async def add(self, exps: List[Experience], step: int) -> Tuple[int, Dict]:
        if len(exps) == 0:
            return 0, {}
        exp_groups = self.group_experiences(exps)
        cnt = 0
        metric_list = []
        tasks = []
        for group_id, group_exps in exp_groups.items():
            group_exps, group_metrics = self.calculate_group_advantage(group_id, group_exps)
            metric_list.append(group_metrics)
            cnt += len(group_exps)
            if len(group_exps) > 0:
                tasks.append(self.writer.write_async(group_exps))
        if tasks:
            await asyncio.gather(*tasks)
        try:
            metrics = gather_metrics(metric_list, "group_advantages")
        except ValueError:
            metrics = {}  # empty metric list causes ValueError, ignore it
        return cnt, metrics


@ADD_STRATEGY.register_module("grpo")
class GRPOAddStrategy(GroupAdvantageStrategy):
    """An example AddStrategy that calculates GRPO advantages."""

    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(writer)
        self.epsilon = epsilon

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)
            for exp in exps:
                score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6}


@ADD_STRATEGY.register_module("opmd")
class OPMDAddStrategy(GroupAdvantageStrategy):
    """An example AddStrategy that calculates OPMD advantages."""

    def __init__(
        self, writer: BufferWriter, opmd_baseline: str = "mean", tau: float = 1.0, **kwargs
    ) -> None:
        super().__init__(writer)
        assert opmd_baseline in [
            "mean",
            "logavgexp",
        ], f"opmd_baseline must be 'mean' or 'logavgexp', got {opmd_baseline}"
        self.opmd_baseline = opmd_baseline
        self.tau = tau

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_baseline = torch.tensor(0.0)
            else:
                group_rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                if self.opmd_baseline == "mean":
                    group_baseline = torch.mean(group_rewards)
                else:
                    group_baseline = self.tau * (
                        torch.logsumexp(group_rewards / self.tau, dim=-1)
                        - torch.log(torch.tensor(len(exps)))
                    )
            for exp in exps:
                score = exp.reward - group_baseline
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()
            metrics = {
                "group_baseline": group_baseline,
            }
        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"opmd_baseline": "mean", "tau": 1.0}


@ADD_STRATEGY.register_module("reward_variance")
class RewardVarianceAddStrategy(AddStrategy):
    """An example AddStrategy that filters experiences based on a reward variance threshold."""

    def __init__(self, writer: BufferWriter, variance_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(writer)
        self.variance_threshold = variance_threshold

    async def add(self, experiences: List[Experience], step: int) -> Tuple[int, Dict]:
        cnt = 0
        metrics = {}
        tasks = []
        with Timer(metrics, "add_strategy_time"):
            grouped_experiences = group_by(experiences, id_type="task")
            for _, group_exps in grouped_experiences.items():
                if len(group_exps) < 2:
                    continue
                rewards = [exp.reward for exp in group_exps]
                variance = np.var(rewards)
                if variance <= self.variance_threshold:
                    continue
                cnt += len(group_exps)
                tasks.append(self.writer.write_async(group_exps))
            if tasks:
                await asyncio.gather(*tasks)
        return cnt, metrics

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
