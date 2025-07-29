from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from trinity.algorithm.sample_strategy.utils import representative_sample
from trinity.buffer import get_buffer_reader
from trinity.common.config import BufferConfig
from trinity.common.experience import Experiences
from trinity.utils.registry import Registry
from trinity.utils.timer import Timer

SAMPLE_STRATEGY = Registry("sample_strategy")


class SampleStrategy(ABC):
    def __init__(self, buffer_config: BufferConfig, **kwargs) -> None:
        self.pad_token_id = buffer_config.pad_token_id

    @abstractmethod
    def sample(self, step: int) -> Tuple[Experiences, Dict, List]:
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


@SAMPLE_STRATEGY.register_module("warmup")
class WarmupSampleStrategy(SampleStrategy):
    """The default sample strategy."""

    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        self.exp_buffer = get_buffer_reader(
            buffer_config.trainer_input.experience_buffer, buffer_config  # type: ignore
        )
        self.sft_warmup_steps = buffer_config.trainer_input.sft_warmup_steps
        if self.sft_warmup_steps > 0 and buffer_config.trainer_input.sft_warmup_dataset is None:
            raise ValueError("sft_warmup_dataset is required when sft_warmup_steps > 0")
        if buffer_config.trainer_input.sft_warmup_dataset is not None:
            self.sft_buffer = get_buffer_reader(
                buffer_config.trainer_input.sft_warmup_dataset, buffer_config
            )
        else:
            self.sft_buffer = None

    def sample(self, step: int, **kwargs) -> Tuple[Experiences, Dict, List]:
        metrics = {}
        with Timer(metrics, "read_time"):
            if step <= self.sft_warmup_steps:
                exp_list = self.sft_buffer.read()
            else:
                exp_list = self.exp_buffer.read()
            repr_samples = representative_sample(exp_list)
        with Timer(metrics, "gather_time"):
            exps = Experiences.gather_experiences(exp_list, self.pad_token_id)  # type: ignore
        return exps, metrics, repr_samples

    @classmethod
    def default_args(cls) -> dict:
        return {}


@SAMPLE_STRATEGY.register_module("default")
class DefaultSampleStrategy(SampleStrategy):
    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        self.exp_buffer = get_buffer_reader(
            buffer_config.trainer_input.experience_buffer, buffer_config  # type: ignore
        )

    def sample(self, step: int, **kwargs) -> Tuple[Any, Dict, List]:
        metrics = {}
        with Timer(metrics, "read_time"):
            exp_list = self.exp_buffer.read()
            repr_samples = representative_sample(exp_list)
        with Timer(metrics, "gather_time"):
            exps = Experiences.gather_experiences(exp_list, self.pad_token_id)  # type: ignore
        return exps, metrics, repr_samples

    @classmethod
    def default_args(cls) -> dict:
        return {}
