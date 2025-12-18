from trinity.algorithm.kl_fn.kl_fn import KLFn
from trinity.utils.registry import Registry

KL_FN: Registry = Registry(
    "kl_fn",
    default_mapping={
        "none": "trinity.algorithm.kl_fn.kl_fn.DummyKLFn",
        "k1": "trinity.algorithm.kl_fn.kl_fn.K1Fn",
        "k2": "trinity.algorithm.kl_fn.kl_fn.K2Fn",
        "k3": "trinity.algorithm.kl_fn.kl_fn.K3Fn",
        "low_var_kl": "trinity.algorithm.kl_fn.kl_fn.LowVarKLFn",
        "abs": "trinity.algorithm.kl_fn.kl_fn.AbsFn",
        "corrected_k3": "trinity.algorithm.kl_fn.kl_fn.CorrectedK3Fn",
    },
)

__all__ = ["KLFn", "KL_FN"]
