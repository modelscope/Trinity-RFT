import copy
from math import ceil
from typing import Dict, List, Tuple

import torch

from trinity.algorithm.sample_strategy.sample_strategy import (
    SAMPLE_STRATEGY,
    SampleStrategy,
)
from trinity.algorithm.sample_strategy.utils import representative_sample
from trinity.buffer import get_buffer_reader
from trinity.common.config import BufferConfig
from trinity.common.experience import CustomField, Experiences
from trinity.utils.timer import Timer


@SAMPLE_STRATEGY.register_module("mix")
class MixSampleStrategy(SampleStrategy):
    """The default sample strategy."""

    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
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

    def sample(self, step: int) -> Tuple[Experiences, Dict, List]:
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
                exp.logprobs = torch.zeros_like(
                    exp.tokens[exp.prompt_length :], dtype=torch.float32
                )
                if exp.info is None:
                    exp.info = {}
                exp.info["is_expert"] = True

            exp_list = usual_exp_list + expert_exp_list
            repr_samples = representative_sample(exp_list)

        with Timer(metrics, "gather_time"):
            exps = Experiences.gather_experiences(
                experiences=exp_list,
                pad_token_id=self.pad_token_id,  # type: ignore [arg-type]
                custom_fields=[
                    CustomField(
                        source_field="is_expert",
                        destination_field="expert_mask",
                        data_type=torch.bool,
                    )
                ],
            )  # type: ignore
        return exps, metrics, repr_samples

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "expert_data_ratio": 0.5,
        }
