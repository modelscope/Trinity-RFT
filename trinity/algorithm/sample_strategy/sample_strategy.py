import copy
from abc import ABC, abstractmethod
from math import ceil
from typing import Any, Dict, List, Tuple

import torch

from trinity.algorithm.sample_strategy.utils import (
    representative_sample,
    to_data_proto,
    to_data_proto_mix,
)
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


@SAMPLE_STRATEGY.register_module("mix")
class MixSampleStrategy(SampleStrategy):
    """The default sample strategy."""

    def __init__(self, buffer_config: BufferConfig, trainer_type: str, **kwargs):
        super().__init__(buffer_config, trainer_type)
        self.expert_data_ratio = kwargs.get("expert_data_ratio", 0.5)
        tot_batch_size = buffer_config.read_batch_size
        expert_batch_size = ceil(self.expert_data_ratio * tot_batch_size)

        # experience buffer
        usual_buffer_config = copy.deepcopy(buffer_config)
        usual_buffer_config.read_batch_size = tot_batch_size - expert_batch_size
        self.usual_exp_buffer = get_buffer_reader(
            buffer_config.trainer_input.experience_buffer, usual_buffer_config  # type: ignore
        )

        if buffer_config.trainer_input.sft_warmup_dataset is None:
            raise ValueError(
                "`buffer_config.trainer_input.sft_warmup_dataset` is required in MIX algorithm"
            )

        # expert experience buffer
        expert_buffer_config = copy.deepcopy(buffer_config)
        expert_buffer_config.read_batch_size = expert_batch_size
        self.expert_exp_buffer = get_buffer_reader(
            buffer_config.trainer_input.sft_warmup_dataset, expert_buffer_config
        )

    def sample(self, step: int) -> Tuple[Any, Dict, List]:
        metrics = {}
        with Timer(metrics, "read_time"):
            usual_exp_list = self.usual_exp_buffer.read()
            for exp in usual_exp_list:
                if exp.info is None:
                    exp.info = {}
                exp.info["is_expert"] = False

            expert_exp_list = self.expert_exp_buffer.read()
            for exp in expert_exp_list:
                exp.reward = 0.0
                exp.logprobs = torch.zeros_like(exp.tokens, dtype=torch.float32)
                if exp.info is None:
                    exp.info = {}
                exp.info["is_expert"] = True

            exp_list = usual_exp_list + expert_exp_list
            repr_samples = representative_sample(exp_list)

        is_expert_mask = torch.tensor([exp.info["is_expert"] for exp in exp_list], dtype=torch.bool)

        with Timer(metrics, "gather_time"):
            exps = Experiences.gather_experiences(exp_list, self.pad_token_id)  # type: ignore

        if self.trainer_type == "verl":
            with Timer(metrics, "convert_time"):
                data = to_data_proto_mix(exps, is_expert_mask)
            return data, metrics, repr_samples
        else:
            raise NotImplementedError(f"backend {self.trainer_type} is not supported")

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "expert_data_ratio": 0.5,
        }
