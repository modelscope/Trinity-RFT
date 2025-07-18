import copy
from math import ceil
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from trinity.algorithm.sample_strategy.sample_strategy import (
    SAMPLE_STRATEGY,
    SampleStrategy,
)
from trinity.algorithm.sample_strategy.utils import representative_sample
from trinity.buffer import get_buffer_reader
from trinity.common.config import BufferConfig
from trinity.common.experience import Experiences
from trinity.utils.timer import Timer


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
    def default_args(cls) -> Dict:
        return {
            "expert_data_ratio": 0.5,
        }


def to_data_proto_mix(experiences: Experiences, is_expert_mask: torch.tensor):
    from verl.trainer.ppo.ray_trainer import DataProto

    attention_mask = experiences.attention_masks
    cumsum = torch.cumsum(attention_mask, dim=-1)
    position_ids = torch.clip(cumsum - 1, 0, None).long()
    batch_dict = {
        "uid": np.array(experiences.group_ids),
        "unique_ids": np.array(experiences.unique_ids),
        "position_ids": position_ids,
        "input_ids": experiences.token_ids.long(),
        "responses": experiences.token_ids[:, experiences.prompt_length :].long(),
        "attention_mask": attention_mask.long(),
        "response_mask": (
            experiences.action_masks[:, experiences.prompt_length :].long()
            if hasattr(experiences, "action_masks") and experiences.action_masks is not None
            else attention_mask[:, experiences.prompt_length :].long()
        ),
        "is_expert_mask": is_expert_mask,
    }
    if experiences.rewards is not None:
        token_level_rewards = torch.zeros(attention_mask.shape, dtype=experiences.rewards.dtype)
        eos_mask_idx = cumsum.argmax(dim=-1)
        token_level_rewards[
            torch.arange(experiences.batch_size), eos_mask_idx
        ] = experiences.rewards
        token_level_rewards = token_level_rewards[:, experiences.prompt_length :]
        batch_dict.update(
            {
                "token_level_scores": token_level_rewards,
                "old_log_probs": experiences.logprobs[:, experiences.prompt_length :],  # type: ignore
            }
        )
    return DataProto.from_single_dict(batch_dict)
