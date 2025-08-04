# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple

import torch

from trinity.algorithm.add_strategy.add_strategy import ADD_STRATEGY, GRPOAddStrategy
from trinity.buffer import BufferWriter
from trinity.common.experience import Experience


@ADD_STRATEGY.register_module("correct_bias")
class CorrectBiasAddStrategy(GRPOAddStrategy):
    """An Addstrategy with GroupAdvantage that corrects for rank bias (https://arxiv.org/pdf/2506.02355)"""

    def __init__(
        self, writer: BufferWriter, epsilon: float = 1e-6, rank_penalty: float = 0.25, **kwargs
    ) -> None:
        super().__init__(writer, epsilon, **kwargs)
        self.rank_penalty = rank_penalty

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)

            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                # correct bias
                old_log_probs = torch.tensor([torch.mean(exp.logprobs, axis=-1) for exp in exps])
                group_ranks = torch.argsort(torch.argsort(old_log_probs))
                group_ranks = group_ranks / len(group_ranks)
                rewards = rewards * (1 - group_ranks * self.rank_penalty)

                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)

            for i, exp in enumerate(exps):
                score = (rewards[i] - group_reward_mean) / (group_reward_std + self.epsilon)
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()

            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }

        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6, "rank_penalty": 0.25}
