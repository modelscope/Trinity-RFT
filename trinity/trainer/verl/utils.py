"""Utils for ccompatibility issues with verl."""

import os
from logging import Logger

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from trinity.common.config import Config
from trinity.common.experience import Experiences


def to_data_proto(experiences: Experiences, logger: Logger) -> DataProto:  # noqa: C901
    """Convert Experiences to verl DataProto."""
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

    if experiences.rewards is not None or experiences.token_level_rewards is not None:
        assert experiences.logprobs is not None
        if experiences.token_level_rewards is not None:
            if experiences.rewards is not None:
                logger.warning(
                    "Both experiences.rewards and experiences.token_level_rewards are provided. "
                    "Using experiences.token_level_rewards."
                )
            token_level_rewards = experiences.token_level_rewards
        else:
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
    if experiences.advantages is not None:
        batch_dict["advantages"] = experiences.advantages
    if experiences.returns is not None:
        batch_dict["returns"] = experiences.returns

    if experiences.multi_modal_inputs is not None:
        batch_size = len(batch_dict["unique_ids"])
        batch_dict["multi_modal_inputs"] = np.array(
            [
                {k: v[i] for k, v in experiences.multi_modal_inputs.items()}
                for i in range(batch_size)
            ],
            dtype=object,
        )

    if experiences.custom_fields:
        for field in experiences.custom_fields:
            if hasattr(experiences, field):
                batch_dict[field] = getattr(experiences, field)
    return DataProto.from_single_dict(batch_dict)


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict:
    """
    Computes various metrics from a batch of data for PPO training.
    Modified from verl.trainer.ppo.metric_utils.compute_data_metrics

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    metrics = {}

    if "token_level_rewards" in batch.batch and "token_level_scores" in batch.batch:
        sequence_score = batch.batch["token_level_scores"].sum(-1)
        sequence_reward = batch.batch["token_level_rewards"].sum(-1)
        metrics.update(
            {
                # score
                "critic/score/mean": torch.mean(sequence_score).detach().item(),
                "critic/score/max": torch.max(sequence_score).detach().item(),
                "critic/score/min": torch.min(sequence_score).detach().item(),
                # reward
                "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
                "critic/rewards/max": torch.max(sequence_reward).detach().item(),
                "critic/rewards/min": torch.min(sequence_reward).detach().item(),
            }
        )

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]
    metrics.update(
        {
            # response length
            "response_length/mean": torch.mean(response_length).detach().item(),
            "response_length/max": torch.max(response_length).detach().item(),
            "response_length/min": torch.min(response_length).detach().item(),
            "response_length/clip_ratio": torch.mean(
                torch.eq(response_length, max_response_length).float()
            )
            .detach()
            .item(),
            # prompt length
            "prompt_length/mean": torch.mean(prompt_length).detach().item(),
            "prompt_length/max": torch.max(prompt_length).detach().item(),
            "prompt_length/min": torch.min(prompt_length).detach().item(),
            "prompt_length/clip_ratio": torch.mean(
                torch.eq(prompt_length, max_prompt_length).float()
            )
            .detach()
            .item(),
        }
    )

    if "advantages" in batch.batch:
        # adv
        advantages = batch.batch["advantages"]
        valid_adv = torch.masked_select(advantages, response_mask)
        metrics.update(
            {
                # adv
                "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
                "critic/advantages/max": torch.max(valid_adv).detach().item(),
                "critic/advantages/min": torch.min(valid_adv).detach().item(),
            }
        )
    if "returns" in batch.batch:
        # returns
        returns = batch.batch["returns"]
        valid_returns = torch.masked_select(returns, response_mask)
        metrics.update(
            {
                "critic/returns/mean": torch.mean(valid_returns).detach().item(),
                "critic/returns/max": torch.max(valid_returns).detach().item(),
                "critic/returns/min": torch.min(valid_returns).detach().item(),
            }
        )

    return metrics


def get_latest_hf_checkpoint_path(config: Config):
    """Get the latest huggingface checkpoint path"""
    if config.trainer.trainer_type != "verl":
        raise ValueError("This function is only for verl trainer.")
    checkpoint_dir = find_latest_ckpt_path(config.checkpoint_job_dir)
    hf_checkpoint_dir = os.path.join(checkpoint_dir, "actor", "huggingface")
    if not os.path.exists(hf_checkpoint_dir):
        raise ValueError(f"No huggingface checkpoint found in {hf_checkpoint_dir}")
    return hf_checkpoint_dir
