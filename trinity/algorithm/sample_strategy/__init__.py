from trinity.algorithm.sample_strategy.sample_strategy import SampleStrategy
from trinity.utils.registry import Registry

SAMPLE_STRATEGY: Registry = Registry(
    "sample_strategy",
    default_mapping={
        "default": "trinity.algorithm.sample_strategy.sample_strategy.DefaultSampleStrategy",
        "warmup": "trinity.algorithm.sample_strategy.sample_strategy.WarmupSampleStrategy",
        "mix": "trinity.algorithm.sample_strategy.mix_sample_strategy.MixSampleStrategy",
    },
)

__all__ = [
    "SAMPLE_STRATEGY",
    "SampleStrategy",
]
