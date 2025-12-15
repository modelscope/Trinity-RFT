from typing import List, Tuple

import numpy as np

from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by


class RewardFilter(ExperienceOperator):
    """
    Filter experiences based on the reward value.

    Note: This filter assumes that the reward is already calculated and stored in the Experience object.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Filter experiences based on reward value."""
        filtered_exps = [exp for exp in exps if exp.reward >= self.threshold]  # type: ignore [operator]
        metrics = {"filtered_count": len(exps) - len(filtered_exps)}
        return filtered_exps, metrics


class RewardSTDFilter(ExperienceOperator):
    """
    Filter experiences based on the standard deviation of rewards within each group.

    Note: This filter assumes that the reward is already calculated and stored in the Experience object.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Filter experiences based on reward std."""
        metrics = {}
        result_exps = []
        original_count = len(exps)
        grouped_experiences = group_by(exps, id_type="task")
        for _, group_exps in grouped_experiences.items():
            if len(group_exps) < 2:
                continue
            rewards = [exp.reward for exp in group_exps]
            variance = np.std(rewards)
            if variance <= self.threshold:
                continue
            result_exps.extend(group_exps)
        final_count = len(result_exps)
        metrics["filtered_count"] = original_count - final_count
        return result_exps, metrics
