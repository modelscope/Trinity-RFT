from logging import Logger
from typing import Any, List, Tuple

import torch
from tinker import types

from trinity.common.experience import Experience, split_dpo_experience_to_single_turn


def to_tinker_input(
    experiences: List[Experience], logger: Logger
) -> Tuple[List[types.Datum], List[types.ModelInput], List[dict]]:
    assert len(experiences) > 0, "No experiences provided."
    if experiences[0].experience_type == "dpo":
        experiences = split_dpo_experience_to_single_turn(experiences)

    batch = []
    batch_input_tokens = []
    model_inputs_list = []
    for exp in experiences:
        tokens = exp.tokens
        input_tokens = tokens.long()
        prompt_length = exp.prompt_length
        total_length = len(tokens)  # type: ignore
        response_length = total_length - prompt_length
        loss_fn_inputs = {
            "weights": torch.concat(
                [
                    torch.zeros(prompt_length - 1, dtype=torch.float32),
                    exp.action_mask.float(),
                ]
            ),
            "target_tokens": input_tokens.tolist()[1:],
        }
        model_inputs = {
            "total_length": total_length,
            "action_mask": exp.action_mask,
        }
        if exp.reward is not None or exp.token_level_reward is not None:
            assert exp.logprobs is not None
            if exp.token_level_reward is not None:
                if exp.reward is not None:
                    logger.warning(
                        "Both exp.rewards and exp.token_level_rewards are provided. "
                        "Using exp.token_level_rewards."
                    )
                token_level_reward = exp.token_level_reward
            else:
                token_level_reward = torch.zeros(response_length, dtype=torch.float32)
                token_level_reward[-1] = exp.reward
            model_inputs.update(
                {
                    "token_level_scores": token_level_reward,
                    "old_logprob": exp.logprobs,
                }
            )
        for attr in ["advantages", "returns", "teacher_logprobs"]:
            if getattr(exp, attr, None) is not None:
                model_inputs[attr] = getattr(exp, attr)
        # TODO: if tinker support multi-modal input, we can add it here
        for custom_field in exp.custom_fields:
            model_inputs[custom_field.destination_field] = torch.tensor(
                exp.info[custom_field.source_field],
                dtype=custom_field.data_type,
            )

        batch.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens.tolist()[:-1]),
                loss_fn_inputs=loss_fn_inputs,
            )
        )
        batch_input_tokens.append(types.ModelInput.from_ints(input_tokens.tolist()))
        model_inputs_list.append(model_inputs)
    return batch, batch_input_tokens, model_inputs_list


def compute_data_metrics(batch: List[dict[str, torch.Tensor]]) -> dict:
    """
    Computes various metrics from a batch of data for PPO training.
    Modified from `verl.trainer.ppo.metric_utils.compute_data_metrics`.

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
            - critic/values/mean, max, min: Statistics about critic values
            - critic/vf_explained_var: Explained variance of the value function
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    metrics = {}

    assert len(batch) > 0, "Batch is empty"

    if "token_level_rewards" in batch[0] and "token_level_scores" in batch[0]:
        sequence_score = torch.tensor([data["token_level_scores"].sum() for data in batch])
        sequence_reward = torch.tensor([data["token_level_rewards"].sum() for data in batch])
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

    response_length = torch.tensor([len(data["action_mask"]) for data in batch]).float()
    token_length = torch.tensor([data["total_length"] for data in batch]).float()
    prompt_length = token_length - response_length
    max_response_length = max(response_length)
    max_prompt_length = max(prompt_length)
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

    if "advantages" in batch[0]:
        valid_adv = torch.concat([data["advantages"] for data in batch])
        metrics.update(
            {
                "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
                "critic/advantages/max": torch.max(valid_adv).detach().item(),
                "critic/advantages/min": torch.min(valid_adv).detach().item(),
            }
        )
    if "returns" in batch[0]:
        valid_returns = torch.concat([data["returns"] for data in batch])
        metrics.update(
            {
                "critic/returns/mean": torch.mean(valid_returns).detach().item(),
                "critic/returns/max": torch.max(valid_returns).detach().item(),
                "critic/returns/min": torch.min(valid_returns).detach().item(),
            }
        )

    return metrics


def compute_timing_metrics(
    batch: List[dict[str, torch.Tensor]], timing_raw: dict[str, float]
) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.
    Modified from `verl.trainer.ppo.metric_utils.compute_timing_metrics`.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    num_overall_tokens = sum(data["total_length"] for data in batch)
    num_response_tokens = sum(len(data["action_mask"]) for data in batch)

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens
            for name in ["ref", "values", "adv", "update_critic", "update_actor"]
        },
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(
    batch: List[dict[str, torch.Tensor]], timing_raw: dict[str, float]
) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.
    Modified from `verl.trainer.ppo.metric_utils.compute_throughout_metrics`.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed and time per step.

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
    """
    total_num_tokens = sum(data["total_length"] for data in batch)
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
    }
