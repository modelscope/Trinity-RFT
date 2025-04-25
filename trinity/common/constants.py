# -*- coding: utf-8 -*-
"""Constants."""
from enum import Enum, EnumMeta

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

    ROLLOUT = "rollout"
    SFT = "sft"
    PPO = "ppo"
    GRPO = "grpo"
    OPMD = "opmd"
    DPO = "dpo"

    def is_rft(self) -> bool:
        """Check if the algorithm is RFT."""
        return self in [
            AlgorithmType.PPO,
            AlgorithmType.GRPO,
            AlgorithmType.OPMD,
        ]

    def is_sft(self) -> bool:
        """Check if the algorithm is SFT."""
        return self == AlgorithmType.SFT

    def is_rollout(self) -> bool:
        """Check if the algorithm is ROLLOUT."""
        return self == AlgorithmType.ROLLOUT

    def is_dpo(self) -> bool:
        """Check if the algorithm is DPO."""
        return self == AlgorithmType.DPO


class MonitorType(CaseInsensitiveEnum):
    """Monitor Type."""

    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
