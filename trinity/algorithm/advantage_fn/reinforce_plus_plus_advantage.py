"""REINFORCE++ advantage computation

Ref: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn import AdvantageFn
from trinity.algorithm.utils import masked_whiten


class REINFORCEPLUSPLUSAdvantageFn(AdvantageFn):
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        """
        Compute advantage for REINFORCE++.
        This implementation is based on the paper: https://arxiv.org/abs/2501.03262

            token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            advantages: `(torch.Tensor)`
                shape: (bs, response_length)
            returns: `(torch.Tensor)`
                shape: (bs, response_length)
        """
        token_level_rewards = exps.batch["token_level_rewards"]
        eos_mask = exps.batch["response_mask"]
        gamma = self.gamma

        with torch.no_grad():
            returns = torch.zeros_like(token_level_rewards)
            running_return = 0

            for t in reversed(range(token_level_rewards.shape[1])):
                running_return = token_level_rewards[:, t] + gamma * running_return
                returns[:, t] = running_return

            advantages = masked_whiten(returns, eos_mask)
            advantages = advantages * eos_mask

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = returns

        metrics = {
            # TODO: add meaningful metrics
        }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "gamma": 1.0,
        }
