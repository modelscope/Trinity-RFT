"""GRPO advantage computation

Adapted from compute_advantage_ppo in original ray_trainer.py
"""

from typing import Dict, Tuple

from verl import DataProto

from trinity.algorithm.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.trainer.verl import core_algos


@ADVANTAGE_FN.register_module("grpo_adv_fn")
class GRPOAdvantageFn(AdvantageFn):
    """GRPO advantage computation"""

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=exps.batch["token_level_rewards"],
            eos_mask=exps.batch["response_mask"],
            index=exps.non_tensor_batch["uid"],
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
