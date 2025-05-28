from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from verl import DataProto

from trinity.trainer.verl import core_algos
from trinity.utils.registry import Registry

ADVANTAGE_FN = Registry("advantage_fn")


class AdvantageFn(ABC):
    @abstractmethod
    def __call__(self, exps: Any, **kwargs: Dict) -> Tuple[Any, Dict]:
        """Calculate advantages from experiences

        Args:
            exps (`DataProto`): The input experiences.
            kwargs (`Dict`): The step-level parameters for calculating advantages.

        Returns:
            `Any`: The experiences with advantages.
            `Dict`: The metrics for logging.
        """


@ADVANTAGE_FN.register("ppo_adv_fn")
class PPOAdvantageFn(AdvantageFn):
    """PPO's GAE advantage computation"""

    def __init__(self, **kwargs):
        self.gamma = kwargs.get("gamma")
        self.lam = kwargs.get("lam")

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Adapted from compute_advantage_ppo in ray_trainer.py"""

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
            "abc": "xyz",  # TODO: add meaningful metrics
        }

        return exps, metrics


@ADVANTAGE_FN.register("grpo_adv_fn")
class GRPOAdvantageFn(AdvantageFn):
    """GRPO advantage computation"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Adapted from compute_advantage_ppo in ray_trainer.py"""

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=exps.batch["token_level_rewards"],
            eos_mask=exps.batch["response_mask"],
            index=exps.non_tensor_batch["uid"],
        )
        exps.batch["advantages"] = advantages
        exps.batch["returns"] = returns

        metrics = {
            "abc": "xyz",  # TODO: add meaningful metrics
        }

        return exps, metrics


@ADVANTAGE_FN.register("reinforceplusplus_adv_fn")
class REINFORCEPLUSPLUSAdvantageFn(AdvantageFn):
    """REINFORCE++ advantage computation"""

    def __init__(self, **kwargs):
        self.gamma = kwargs.get("gamma")

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Adapted from compute_advantage_ppo in ray_trainer.py"""

        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=exps.batch["token_level_rewards"],
            eos_mask=exps.batch["response_mask"],
            gamma=self.gamma,
        )
        exps.batch["advantages"] = advantages
        exps.batch["returns"] = returns

        metrics = {
            "abc": "xyz",  # TODO: add meaningful metrics
        }

        return exps, metrics


@ADVANTAGE_FN.register("remax_adv_fn")
class REMAXAdvantageFn(AdvantageFn):
    """REMAX advantage computation"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Adapted from compute_advantage_ppo in ray_trainer.py"""

        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=exps.batch["token_level_rewards"],
            reward_baselines=exps.batch["reward_baselines"],
            eos_mask=exps.batch["response_mask"],
        )
        exps.batch["advantages"] = advantages
        exps.batch["returns"] = returns

        metrics = {
            "abc": "xyz",  # TODO: add meaningful metrics
        }

        return exps, metrics


@ADVANTAGE_FN.register("rloo_adv_fn")
class RLOOAdvantageFn(AdvantageFn):
    """RLOO advantage computation"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Adapted from compute_advantage_ppo in ray_trainer.py"""

        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=exps.batch["token_level_rewards"],
            eos_mask=exps.batch["response_mask"],
            index=exps.non_tensor_batch["uid"],
        )
        exps.batch["advantages"] = advantages
        exps.batch["returns"] = returns

        metrics = {
            "abc": "xyz",  # TODO: add meaningful metrics
        }

        return exps, metrics


@ADVANTAGE_FN.register("opmd_adv_fn")
class OPMDAdvantageFn(AdvantageFn):
    """OPMD advantage computation"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Adapted from compute_advantage_opmd in ray_trainer.py"""

        advantages, returns = core_algos.compute_opmd_outcome_advantage(
            token_level_rewards=exps.batch["token_level_rewards"],
            eos_mask=exps.batch["response_mask"],
            # TODO: check consistency with exps.batch["attention_mask"][:, -response_length:] in original implementation
            index=exps.non_tensor_batch["uid"],
            opmd_baseline="mean",
            tau=1.0,
        )
        exps.batch["advantages"] = advantages
        exps.batch["returns"] = returns

        metrics = {
            "abc": "xyz",  # TODO: add meaningful metrics
        }

        return exps, metrics
