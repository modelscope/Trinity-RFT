from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import EntropyLossFn
from trinity.utils.registry import Registry

ENTROPY_LOSS_FN: Registry = Registry(
    "entropy_loss_fn",
    default_mapping={
        "default": "trinity.algorithm.entropy_loss_fn.entropy_loss_fn.DefaultEntropyLossFn",
        "mix": "trinity.algorithm.entropy_loss_fn.entropy_loss_fn.MixEntropyLossFn",
        "none": "trinity.algorithm.entropy_loss_fn.entropy_loss_fn.DummyEntropyLossFn",
    },
)

__all__ = [
    "EntropyLossFn",
    "ENTROPY_LOSS_FN",
]
