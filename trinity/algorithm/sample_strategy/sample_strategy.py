from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from trinity.algorithm.sample_strategy.utils import representative_sample
from trinity.buffer import get_buffer_reader
from trinity.common.config import BufferConfig
from trinity.common.experience import Experiences
from trinity.utils.annotations import Deprecated
from trinity.utils.registry import Registry
from trinity.utils.timer import Timer

SAMPLE_STRATEGY = Registry("sample_strategy")


class SampleStrategy(ABC):
    def __init__(self, buffer_config: BufferConfig, **kwargs) -> None:
        self.pad_token_id = buffer_config.pad_token_id

    @abstractmethod
    async def sample(self, step: int) -> Tuple[Experiences, Dict, List]:
        """Sample data from buffer.

        Args:
            step (`int`): The step number of current step.

        Returns:
            `Experiences`: The sampled Experiences data.
            `Dict`: Metrics for logging.
            `List`: Representative data for logging.
        """

    @classmethod
    @abstractmethod
    def default_args(cls) -> dict:
        """Get the default arguments of the sample strategy."""


@SAMPLE_STRATEGY.register_module("default")
class DefaultSampleStrategy(SampleStrategy):
    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        self.exp_buffer = get_buffer_reader(buffer_config.trainer_input.experience_buffer)  # type: ignore[arg-type]

    async def sample(self, step: int, **kwargs) -> Tuple[Any, Dict, List]:
        metrics = {}
        with Timer(metrics, "time/read_experience"):
            exp_list = await self.exp_buffer.read_async()
            repr_samples = representative_sample(exp_list)
        with Timer(metrics, "time/gather_experience"):
            exps = Experiences.gather_experiences(exp_list, self.pad_token_id)  # type: ignore
        return exps, metrics, repr_samples

    @classmethod
    def default_args(cls) -> dict:
        return {}


@Deprecated
@SAMPLE_STRATEGY.register_module("warmup")
class WarmupSampleStrategy(DefaultSampleStrategy):
    """The warmup sample strategy.
    Deprecated, keep this class for backward compatibility only.
    Please use DefaultSampleStrategy instead."""

    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
