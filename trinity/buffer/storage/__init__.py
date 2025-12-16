from trinity.buffer.storage.queue import PriorityFunction
from trinity.utils.registry import Registry

PRIORITY_FUNC = Registry(
    "priority_fn",
    default_mapping={
        "linear_decay": "trinity.buffer.storage.queue.LinearDecayPriority",
        "decay_limit_randomization": "trinity.buffer.storage.queue.LinearDecayUseCountControlPriority",
    },
)

__all__ = [
    "PriorityFunction",
    "PRIORITY_FUNC",
]
