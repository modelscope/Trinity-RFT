# -*- coding: utf-8 -*-
"""Constants."""
from enum import Enum, EnumMeta

from trinity.utils.log import get_logger

logger = get_logger(__name__)

# names

ROLLOUT_WEIGHT_SYNC_GROUP_NAME = "rollout_weight_sync"


# enumerate types


class CaseInsensitiveEnumMeta(EnumMeta):
    def __getitem__(cls, name):
        return super().__getitem__(name.upper())

    def __getattr__(cls, name):
        if not name.startswith("_"):
            return cls[name.upper()]
        return super().__getattr__(name)


class CaseInsensitiveEnum(Enum, metaclass=CaseInsensitiveEnumMeta):
    pass


class PromptType(CaseInsensitiveEnum):
    """Prompt Type."""

    MESSAGES = "messages"  # prompt+response: message list
    CHATPAIR = "chatpair"  # prompt: message list, response: message list
    PLAINTEXT = "plaintext"  # prompt: plaintext, response: plaintext


class TaskType(Enum):
    """Task Type."""

    EXPLORE = 0
    EVAL = 1


class ReadStrategy(CaseInsensitiveEnum):
    """Pop Strategy."""

    DEFAULT = None
    FIFO = "FIFO"
    RANDOM = "RANDOM"
    LRU = "LRU"
    LFU = "LFU"
    PRIORITY = "PRIORITY"


class StorageType(CaseInsensitiveEnum):
    """Storage Type."""

    SQL = "sql"
    QUEUE = "queue"
    FILE = "file"


class AlgorithmType(CaseInsensitiveEnum):
    """Algorithm Type."""

    SFT = "sft"
    PPO = "ppo"
    GRPO = "grpo"
    OPMD = "opmd"
    PAIRWISE_OPMD = "pairwise_opmd"
    DPO = "dpo"

    def is_rft(self) -> bool:
        """Check if the algorithm is RFT."""
        return self in [
            AlgorithmType.PPO,
            AlgorithmType.GRPO,
            AlgorithmType.OPMD,
            AlgorithmType.PAIRWISE_OPMD,
        ]

    def is_sft(self) -> bool:
        """Check if the algorithm is SFT."""
        return self == AlgorithmType.SFT

    def is_dpo(self) -> bool:
        """Check if the algorithm is DPO."""
        return self == AlgorithmType.DPO


class MonitorType(CaseInsensitiveEnum):
    """Monitor Type."""

    WANDB = "wandb"
    TENSORBOARD = "tensorboard"


class SyncMethodEnumMeta(CaseInsensitiveEnumMeta):
    def __call__(cls, value, *args, **kwargs):
        if value == "online":
            logger.warning("SyncMethod `online` is deprecated, use `nccl` instead.")
            value = "nccl"
        elif value == "offline":
            logger.warning("SyncMethod `offline` is deprecated, use `checkpoint` instead.")
            value = "checkpoint"
        try:
            return super().__call__(value, *args, **kwargs)
        except Exception as e:
            logger.warning("Error parsing SyncMethod:", e)
            raise ValueError(f"Invalid SyncMethod: {value}")


class SyncMethod(CaseInsensitiveEnum, metaclass=SyncMethodEnumMeta):
    """Sync Method."""

    NCCL = "nccl"
    CHECKPOINT = "checkpoint"
