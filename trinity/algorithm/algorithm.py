# -*- coding: utf-8 -*-
"""Algorithm classes."""

from abc import ABC, ABCMeta, abstractmethod
from typing import Dict

from trinity.common.config import Config
from trinity.common.constants import SyncMethod
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

logger = get_logger(__name__)

ALGORITHM_TYPE = Registry("algorithm")


class ConstantMeta(ABCMeta):
    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise AttributeError(f"{name} is already defined in {cls.__name__}")
        return super().__setattr__(name, value)


class AlgorithmType(ABC, metaclass=ConstantMeta):
    use_critic: bool  # whether to use critic model

    use_reference: bool  # whether to use reference model

    compute_advantage_in_trainer: bool  # whether to compute advantage in trainer
    # For algorithms that rely on experience grouping,
    # we recommend set this value to False

    can_balance_batch: bool  # balance batch in trainer

    schema: str  # schema of training data

    @classmethod
    @abstractmethod
    def default_config(cls) -> Dict:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls._name

    @classmethod
    def check_config(cls, config: Config) -> None:
        pass


@ALGORITHM_TYPE.register_module("sft")
class SFTAlgorithm(AlgorithmType):
    """SFT Algorithm."""

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "sft"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "sample_strategy": "default",
            "policy_loss_fn": "sft",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


@ALGORITHM_TYPE.register_module("ppo")
class PPOAlgorithm(AlgorithmType):
    """PPO Algorithm."""

    use_critic: bool = True
    use_reference: bool = True
    compute_advantage_in_trainer: bool = True
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 1,
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "advantage_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


@ALGORITHM_TYPE.register_module("grpo")
class GRPOAlgorithm(AlgorithmType):
    """GRPO algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "grpo",
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


@ALGORITHM_TYPE.register_module("opmd")
class OPMDAlgorithm(AlgorithmType):
    """OPMD algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "opmd",
            "sample_strategy": "default",
            "policy_loss_fn": "opmd",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


@ALGORITHM_TYPE.register_module("asymre")
class AsymREAlgorithm(AlgorithmType):
    """AsymRE algorithm."""

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "sample_strategy": "default",
            "policy_loss_fn": "opmd",
            "advantage_fn": "asymre",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


@ALGORITHM_TYPE.register_module("dpo")
class DPOAlgorithm(AlgorithmType):
    """DPO algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = False
    schema: str = "dpo"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "sample_strategy": "default",
            "policy_loss_fn": "dpo",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }

    @classmethod
    def check_config(cls, config: Config) -> None:
        if config.mode == "train":
            if (
                config.buffer.trainer_input.experience_buffer is None
                or not config.buffer.trainer_input.experience_buffer.path
            ):
                raise ValueError(
                    "`buffer.trainer_input.experience_buffer.path` is required when `algorithm.algorithm_type == dpo`"
                )
        elif config.mode in ["both", "explore"]:
            raise ValueError(f"DPO does not support `{config.mode}` mode")

        if config.synchronizer.sync_method != SyncMethod.CHECKPOINT:
            config.synchronizer.sync_method = SyncMethod.CHECKPOINT
            logger.warning(
                "DPO only supports checkpoint synchronization, set `synchronizer.sync_method` to `checkpoint`."
            )
        if config.algorithm.repeat_times != 2:
            config.algorithm.repeat_times = 2  # Fake repeat times
        if config.algorithm.kl_loss_fn in {"none", None}:
            config.algorithm.kl_loss_fn = "k2"
            logger.warning("DPO must use KL loss. Set `algorithm.kl_loss_fn` to `k2`")


@ALGORITHM_TYPE.register_module("topr")
class TOPRAlgorithm(AlgorithmType):
    """TOPR algorithm. See https://arxiv.org/pdf/2503.14286v1"""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "reinforce",  # or simply use grpo
            "sample_strategy": "default",
            "policy_loss_fn": "topr",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


@ALGORITHM_TYPE.register_module("cispo")
class CISPOAlgorithm(AlgorithmType):
    """CISPO algorithm. See https://arxiv.org/abs/2506.13585"""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "grpo",
            "sample_strategy": "default",
            "policy_loss_fn": "cispo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }


@ALGORITHM_TYPE.register_module("mix")
class MIXAlgorithm(AlgorithmType):
    """MIX algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    use_rollout: bool = True
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 8,
            "advantage_fn": "grpo",
            "policy_loss_fn": "mix",
            "sample_strategy": "mix",
            "entropy_loss_fn": "mix",
        }


@ALGORITHM_TYPE.register_module("mix_chord")
class MIXCHORDAlgorithm(AlgorithmType):
    """MIX algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    use_rollout: bool = True
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 8,
            "policy_loss_fn": "mix_chord",
            "advantage_fn": "grpo",
            "sample_strategy": "mix",
            "entropy_loss_fn": "mix",
        }


@ALGORITHM_TYPE.register_module("raft")
class RAFTAlgorithm(AlgorithmType):
    """RAFT Algorithm.
    This algorithm is conceptually similar to Supervised Fine-Tuning (SFT)
    but is designed to work with `experience` schema from rollouts.
    """

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "sample_strategy": "default",
            "policy_loss_fn": "sft",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


@ALGORITHM_TYPE.register_module("sppo")
class sPPOAlgorithm(AlgorithmType):
    """sPPO Algorithm."""

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "sample_strategy": "default",
            "policy_loss_fn": "sppo",
            "advantage_fn": "opmd",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


@ALGORITHM_TYPE.register_module("rec")
class RECAlgorithm(AlgorithmType):
    """REC Algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "sample_strategy": "default",
            "policy_loss_fn": "rec",
            "advantage_fn": "rec",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }


@ALGORITHM_TYPE.register_module("multi_step_grpo")
class MultiStepGRPOAlgorithm(AlgorithmType):
    """Multi-Step GRPO Algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 8,
            "advantage_fn": "step_wise_grpo",
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }
