"""Reinforce advantage computation"""

from typing import Dict, List, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import GroupAdvantage
from trinity.common.experience import Experience, group_by


class REINFORCEGroupAdvantage(GroupAdvantage):
    """Reinforce Group Advantage computation"""

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
            group_reward_mean = torch.mean(rewards)
            for exp in exps:
                score = torch.tensor(exp.reward, dtype=torch.float32)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": group_reward_mean.item(),
            }
        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {}
