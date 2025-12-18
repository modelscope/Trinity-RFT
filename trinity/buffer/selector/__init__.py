from trinity.buffer.selector.selector import BaseSelector
from trinity.utils.registry import Registry

SELECTORS = Registry(
    "selectors",
    default_mapping={
        "sequential": "trinity.buffer.selector.selector.SequentialSelector",
        "shuffle": "trinity.buffer.selector.selector.ShuffleSelector",
        "random": "trinity.buffer.selector.selector.RandomSelector",
        "offline_easy2hard": "trinity.buffer.selector.selector.OfflineEasy2HardSelector",
        "difficulty_based": "trinity.buffer.selector.selector.DifficultyBasedSelector",
    },
)

__all__ = [
    "BaseSelector",
    "SELECTORS",
]
