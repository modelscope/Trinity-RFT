"""REMAX advantage computation

Ref: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn import AdvantageFn


class REMAXAdvantageFn(AdvantageFn):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        """
        Compute advantage for ReMax, operating only on Outcome reward
        (with only one scalar reward for each response).
        This implementation is based on the paper: https://arxiv.org/abs/2310.10505

            token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            reward_baselines: `(torch.Tensor)`
                shape: (bs,)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            advantages: `(torch.Tensor)`
                shape: (bs, response_length)
            returns: `(torch.Tensor)`
                shape: (bs, response_length)
        """
        token_level_rewards = exps.batch["token_level_rewards"]
        reward_baselines = exps.batch["reward_baselines"]
        eos_mask = exps.batch["response_mask"]

        response_length = token_level_rewards.shape[-1]
        token_level_rewards.sum(dim=-1)

        with torch.no_grad():
            returns = (
                (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
            )
            advantages = (
                returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask
            )

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = returns

        metrics = {
            # TODO: add meaningful metrics
        }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {}
