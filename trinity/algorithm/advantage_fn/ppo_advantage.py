"""PPO's GAE advantage computation

Adapted from compute_advantage_ppo in original ray_trainer.py
"""

from typing import Dict, Tuple

from verl import DataProto

from trinity.algorithm.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.trainer.verl import core_algos


@ADVANTAGE_FN.register_module("ppo_adv_fn")
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
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=exps.batch["token_level_rewards"],
            values=exps.batch["values"],
            eos_mask=exps.batch["response_mask"],
            gamma=self.gamma,
            lam=self.lam,
        )
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
