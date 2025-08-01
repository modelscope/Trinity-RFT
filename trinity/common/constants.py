# -*- coding: utf-8 -*-
"""Constants."""
from enum import Enum, EnumMeta

from trinity.utils.log import get_logger

logger = get_logger(__name__)

# names

EXPLORER_NAME = "explorer"
TRAINER_NAME = "trainer"

ROLLOUT_WEIGHT_SYNC_GROUP_NAME = "rollout_weight_sync"


# enumerate types


class CaseInsensitiveEnumMeta(EnumMeta):
    def __getitem__(cls, name):
        return super().__getitem__(name.upper())

    def __getattr__(cls, name):
        if not name.startswith("_"):
            return cls[name.upper()]
        return super().__getattr__(name)

    def __call__(cls, value, *args, **kwargs):
        return super().__call__(value.lower(), *args, **kwargs)


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
    FIFO = "fifo"
    RANDOM = "random"
    LRU = "lru"
    LFU = "lfu"
    PRIORITY = "priority"


class StorageType(CaseInsensitiveEnum):
    """Storage Type."""

    SQL = "sql"
    QUEUE = "queue"
    FILE = "file"


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
    MEMORY = "memory"


class RunningStatus(Enum):
    """Running status of explorer and trainer."""

    RUNNING = "running"
    REQUIRE_SYNC = "require_sync"
    WAITING_SYNC = "waiting_sync"
    STOPPED = "stopped"


class DataProcessorPipelineType(Enum):
    """Data processor pipeline type."""

    EXPERIENCE = "experience_pipeline"
    TASK = "task_pipeline"


class OpType(Enum):
    """Operator type for reward shaping."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"


class SyncStyle(CaseInsensitiveEnum):
    FIXED = "fixed"
    DYNAMIC_BY_TRAINER = "dynamic_by_trainer"
    DYNAMIC_BY_EXPLORER = "dynamic_by_explorer"
