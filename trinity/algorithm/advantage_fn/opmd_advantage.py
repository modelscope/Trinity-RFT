"""OPMD advantage computation

Adapted from compute_advantage_opmd in original ray_trainer.py
"""

from typing import Dict, Tuple

from verl import DataProto

from trinity.algorithm.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.trainer.verl import core_algos


@ADVANTAGE_FN.register_module("opmd_adv_fn")
class OPMDAdvantageFn(AdvantageFn):
    """OPMD advantage computation"""

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        advantages, returns = core_algos.compute_opmd_outcome_advantage(
            token_level_rewards=exps.batch["token_level_rewards"],
            eos_mask=exps.batch["response_mask"],
            # TODO (yanxi): check consistency with exps.batch["attention_mask"][:, -response_length:] in original implementation
            index=exps.non_tensor_batch["uid"],
            opmd_baseline="mean",
            tau=1.0,
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
