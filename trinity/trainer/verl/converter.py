"""Convert Experiences to verl.DataProto."""

import numpy as np
import torch
from verl import DataProto

from trinity.common.experience import Experiences


def to_data_proto(experiences: Experiences) -> DataProto:
    attention_mask = experiences.attention_masks
    cumsum = torch.cumsum(attention_mask, dim=-1)
    position_ids = torch.clip(cumsum - 1, 0, None).long()
    batch_dict = {
        "uid": np.array([eid.tid for eid in experiences.eids]),
        "unique_ids": np.array([eid.uid for eid in experiences.eids]),
        "position_ids": position_ids,
        "input_ids": experiences.tokens.long(),
        "responses": experiences.tokens[:, experiences.prompt_length :].long(),
        "attention_mask": attention_mask.long(),
        "response_mask": (
            experiences.action_masks.long()
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
                "old_log_probs": experiences.logprobs,  # type: ignore
            }
        )
    if experiences.custom_fields:
        for field in experiences.custom_fields:
            if hasattr(experiences, field):
                batch_dict[field] = getattr(experiences, field)
    return DataProto.from_single_dict(batch_dict)
