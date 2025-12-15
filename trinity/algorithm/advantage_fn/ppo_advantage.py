"""PPO's GAE advantage computation

Ref: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn
from trinity.algorithm.utils import masked_whiten


class PPOAdvantageFn(AdvantageFn):
    def __init__(
        self,
        gamma: float = 1.0,
        lam: float = 1.0,
    ) -> None:
        self.gamma = gamma
        self.lam = lam

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        """
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
        """
        token_level_rewards = exps.batch["token_level_rewards"]
        values = exps.batch["values"]
        eos_mask = exps.batch["response_mask"]
        gamma = self.gamma
        lam = self.lam

        with torch.no_grad():
            lastgaelam = 0
            advantages_reversed = []
            gen_len = token_level_rewards.shape[-1]

            # values = values * eos_mask TODO: may use in multi-turn
            for t in reversed(range(gen_len)):
                nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

                lastgaelam = delta + gamma * lam * lastgaelam
                # lastgaelam = torch.where(  # TODO: may use in multi-turn
                #     eos_mask[:, t] == 1, delta + gamma * lam * lastgaelam, lastgaelam
                # )
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)

            returns = advantages + values
            advantages = masked_whiten(advantages, eos_mask)

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
            "lam": 1.0,
        }
