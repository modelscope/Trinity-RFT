from abc import ABC, abstractmethod
from typing import Dict

from trinity.common.experience import Experience, Experiences
from trinity.utils.registry import Registry

ALGORITHM = Registry("algorithm")


class Algorithm(ABC):
    use_critic: bool
    use_reference: bool
    use_advantage: bool
    can_balance_batch: bool

    @classmethod
    def gather_experience(cls, exps: list[Experience], pad_token_id: int = 0) -> Experiences:
        return Experiences.gather_experiences(exps, pad_token_id)

    @abstractmethod
    @classmethod
    def get_default_config(cls) -> Dict:
        pass


@ALGORITHM.register_module("sft")
class SFTAlgorithm(Algorithm):
    """SFT Algorithm."""

    use_critic: bool = False
    use_reference: bool = False
    use_advantage: bool = False
    can_balance_batch: bool = True

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "policy_loss_fn": "sft",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "basic",
        }


@ALGORITHM.register_module("ppo")
class PPOAlgorithm(Algorithm):
    """PPO Algorithm."""

    use_critic: bool = True
    use_reference: bool = True
    use_advantage: bool = True
    can_balance_batch: bool = True

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "repeat_times": 1,
            "policy_loss_fn": "ppo",
            "advantage_fn": "ppo",
            "kl_penalty_fn": "k3",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "basic",
        }


@ALGORITHM.register_module("grpo")
class GRPOAlgorithm(Algorithm):
    """GRPO algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    use_advantage: bool = True
    can_balance_batch: bool = True

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "policy_loss_fn": "ppo",
            "advantage_fn": "grpo",
            "kl_penalty_fn": "k3",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "basic",
        }


@ALGORITHM.register_module("opmd")
class OPMDAlgorithm(Algorithm):
    """OPMD algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    use_advantage: bool = True
    can_balance_batch: bool = True

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "policy_loss_fn": "opmd",
            "advantage_fn": "opmd",
            "kl_penalty_fn": "k3",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "basic",
        }


@ALGORITHM.register_module("dpo")
class DPOAlgorithm(Algorithm):
    """DPO algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    use_advantage: bool = False
    can_balance_batch: bool = False

    @classmethod
    def gather_experience(cls, exps: list[Experience], pad_token_id: int = 0) -> Experiences:
        return Experiences.gather_dpo_experiences(exps, pad_token_id)

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "repeat_times": 2,  # fake repeat times
            "policy_loss_fn": "dpo",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "basic",
        }
