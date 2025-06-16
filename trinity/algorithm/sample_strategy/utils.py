import random
from typing import List

import numpy as np
import torch
from verl.trainer.ppo.ray_trainer import DataProto

from trinity.common.experience import Experience, Experiences


def to_data_proto(experiences: Experiences) -> DataProto:
    attention_mask = experiences.attention_masks
    cumsum = torch.cumsum(attention_mask, dim=-1)
    position_ids = torch.clip(cumsum - 1, 0, None).long()
    batch_dict = {
        "uid": np.array(experiences.run_ids),
        "position_ids": position_ids,
        "input_ids": experiences.tokens.long(),
        "responses": experiences.tokens[:, experiences.prompt_length :].long(),
        "attention_mask": attention_mask.long(),
        "response_mask": (
            experiences.action_masks[:, experiences.prompt_length :].long()
            if hasattr(experiences, "action_masks") and experiences.action_masks is not None
            else attention_mask[:, experiences.prompt_length :].long()
        ),
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


def to_data_proto_mix(experiences: Experiences, is_expert_mask: torch.tensor) -> DataProto:
    attention_mask = experiences.attention_masks
    cumsum = torch.cumsum(attention_mask, dim=-1)
    position_ids = torch.clip(cumsum - 1, 0, None).long()
    batch_dict = {
        "uid": np.array(experiences.run_ids),
        "position_ids": position_ids,
        "input_ids": experiences.tokens.long(),
        "responses": experiences.tokens[:, experiences.prompt_length :].long(),
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


def representative_sample(experiences: List[Experience]) -> List[dict]:
    if experiences[0].reward is None:
        sample = random.choice(experiences)
        return [
            {
                "prompt": sample.prompt_text,
                "response": sample.response_text,
            }
        ]
    samples = []
    min_reward_sample = None
    max_reward_sample = None
    for exp in experiences:
        if exp.reward is None:
            continue
        if min_reward_sample is None or exp.reward < min_reward_sample.reward:
            min_reward_sample = exp
        if max_reward_sample is None or exp.reward > max_reward_sample.reward:
            max_reward_sample = exp
    if min_reward_sample is not None:
        samples.append(
            {
                "prompt": min_reward_sample.prompt_text,
                "response": min_reward_sample.response_text,
                "reward": min_reward_sample.reward,
            }
        )
    if max_reward_sample is not None:
        samples.append(
            {
                "prompt": max_reward_sample.prompt_text,
                "response": max_reward_sample.response_text,
                "reward": max_reward_sample.reward,
            }
        )
    return samples
