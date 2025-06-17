from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from trinity.algorithm.sample_strategy.utils import representative_sample, to_data_proto
from trinity.buffer import get_buffer_reader
from trinity.common.config import BufferConfig
from trinity.common.experience import Experiences
from trinity.utils.registry import Registry
from trinity.utils.timer import Timer

SAMPLE_STRATEGY = Registry("sample_strategy")


class SampleStrategy(ABC):
    def __init__(self, buffer_config: BufferConfig, trainer_type: str, **kwargs):
        self.pad_token_id = buffer_config.pad_token_id
        self.trainer_type = trainer_type

    @abstractmethod
    def sample(self, step: int) -> Tuple[Any, Dict, List]:
        """Sample experiences from buffer.

        Args:
            step (`int`): The step number of current step.

        Returns:
            `Any`: The sampled experiences.
            `Dict`: Metrics for logging.
            `List`: Representative experiences for logging.
        """

    @classmethod
    def default_args(cls) -> dict:
        return {}


@SAMPLE_STRATEGY.register_module("warmup")
class WarmupSampleStrategy(SampleStrategy):
    """The default sample strategy."""

    def __init__(self, buffer_config: BufferConfig, trainer_type: str, **kwargs):
        super().__init__(buffer_config, trainer_type)
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

    def sample(self, step: int, **kwargs) -> Tuple[Any, Dict, List]:
        metrics = {}
        with Timer(metrics, "read_time"):
            if step <= self.sft_warmup_steps:
                exp_list = self.sft_buffer.read()
            else:
                exp_list = self.exp_buffer.read()
            repr_samples = representative_sample(exp_list)
        with Timer(metrics, "gather_time"):
            exps = Experiences.gather_experiences(exp_list, self.pad_token_id)  # type: ignore
        if self.trainer_type == "verl":
            with Timer(metrics, "convert_time"):
                data = to_data_proto(exps)
            return data, metrics, repr_samples
        else:
            raise NotImplementedError(f"backend {self.trainer_type} is not supported")


@SAMPLE_STRATEGY.register_module("default")
class DefaultSampleStrategy(SampleStrategy):
    def __init__(self, buffer_config: BufferConfig, trainer_type: str, **kwargs):
        super().__init__(buffer_config, trainer_type)
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
        if self.trainer_type == "verl":
            with Timer(metrics, "convert_time"):
                data = to_data_proto(exps)
            return data, metrics, repr_samples
        else:
            raise NotImplementedError(f"backend {self.trainer_type} is not supported")


@SAMPLE_STRATEGY.register_module("dpo")
class DPOSampleStrategy(WarmupSampleStrategy):
    def sample(self, step: int, **kwargs) -> Tuple[Any, Dict, List]:
        metrics = {}
        with Timer(metrics, "read_time"):
            if step <= self.sft_warmup_steps:
                exp_list = self.sft_buffer.read()
            else:
                exp_list = self.exp_buffer.read()
            repr_samples = representative_sample(exp_list)
        with Timer(metrics, "gather_time"):
            exps = Experiences.gather_dpo_experiences(exp_list, pad_token_id=self.pad_token_id)  # type: ignore
        if self.trainer_type == "verl":
            with Timer(metrics, "convert_time"):
                data = to_data_proto(exps)
            return data, metrics, repr_samples
        else:
            raise NotImplementedError(f"backend {self.trainer_type} is not supported")
