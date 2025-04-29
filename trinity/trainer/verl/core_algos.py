# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Modified from core_algos.py
"""

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import verl.utils.torch_functional as verl_F

from trinity.common.constants import AlgorithmType


class KLController(ABC):
    @abstractmethod
    def update(self, current_kl, n_steps):
        """update value"""


class AdaptiveKLController(KLController):
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController(KLController):
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_config):
    if kl_config.type == "fixed":
        return FixedKLController(kl_coef=kl_config.kl_coef)
    elif kl_config.type == "adaptive":
        assert kl_config.horizon > 0, f"horizon must be larger than 0. Got {kl_config.horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_config.kl_coef,
            target_kl=kl_config.target_kl,
            horizon=kl_config.horizon,
        )
    else:
        raise ValueError("Unknown kl_ctrl type")


def compute_opmd_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    opmd_baseline: str = "mean",
    tau: float = 1.0,
):
    """Modified from compute_grpo_outcome_advantage

    Compute advantage for OPMD, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2baseline = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2baseline[idx] = torch.tensor(0.0)
                # TODO: consider id2baseline[idx] = id2score[idx] (so that this sample won't take effect?)
            elif len(id2score[idx]) > 1:
                if opmd_baseline == "mean":
                    id2baseline[idx] = torch.mean(torch.tensor(id2score[idx]))
                elif opmd_baseline == "logavgexp":
                    rewards_tensor = torch.tensor(id2score[idx])
                    # NOTE: we use the fact that logavgexp(x) = logsumexp(x) - log(len(x)).
                    # Hopefully the logsumexp calculation is numerically stable (as claimed by PyTorch's doc)
                    # in cases where tau is small...
                    id2baseline[idx] = tau * (
                        torch.logsumexp(rewards_tensor / tau, dim=-1)
                        - torch.log(torch.tensor(len(id2score[idx])))
                    )
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2baseline[index[i]]
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # values = values * eos_mask TODO: may use in multi-turn
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

            lastgaelam = delta + gamma * lam * lastgaelam
            # lastgaelam = torch.where(  # TODO: may use in multi-turn
            #     eos_mask[:, t] == 1, delta + gamma * lam * lastgaelam, lastgaelam
            # )
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, eos_mask: torch.Tensor
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, eos_mask, **kwargs):
    """Compute policy loss for PPO / OPMD / pairwise OPMD"""

    algorithm_type: AlgorithmType = kwargs.get("algorithm_type", AlgorithmType.PPO)

    if algorithm_type == AlgorithmType.OPMD:
        advantages = kwargs.get("advantages")
        tau = kwargs.get("tau")
        return compute_policy_loss_opmd(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=eos_mask,
            tau=tau,
        )

    elif algorithm_type == AlgorithmType.PAIRWISE_OPMD:
        token_level_scores = kwargs.get("token_level_scores")
        index = kwargs.get("index")
        tau = kwargs.get("tau")
        return compute_policy_loss_pairwise_opmd(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            token_level_scores=token_level_scores,
            eos_mask=eos_mask,
            index=index,
            tau=tau,
        )

    elif algorithm_type.is_rft():
        advantages = kwargs.get("advantages")
        cliprange = kwargs.get("cliprange")
        return compute_policy_loss_ppo(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=eos_mask,
            cliprange=cliprange,
        )

    else:
        raise NotImplementedError(f"Get invalid algorithm_type '{algorithm_type}'.")


def compute_policy_loss_dpo(
    log_prob, ref_log_prob, eos_mask, loss_type="sigmoid", beta=0.1, label_smoothing=0.0
):
    """Compute policy loss for DPO (Direct Preference Optimization)

    Ref: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L918

    Args:
        log_prob: `(torch.Tensor)`
            The log probabilities of the chosen responses from the policy model.
        ref_log_prob: `(torch.Tensor)`
            The log probabilities of the chosen responses from the reference model.
        loss_type: `(str)`
            Default: "sigmoid"
            The type of loss function to use.
        beta: `(float)`
            Default: 0.1
            A temperature parameter that controls the sharpness of the preference signal.
            Higher values make the loss more sensitive to small differences in log probabilities.
        label_smoothing: `(float)`
            Default: 0.0
            A parameter to encode uncertainty about the labels. Adds a small amount of smoothing to the loss
            to avoid overconfident predictions.

    Returns:
        dpo_loss: `a scalar torch.Tensor`
        chosen_diff: `(torch.Tensor)`
        rejected_diff: `(torch.Tensor)`
    """
    # log_prob: chosen, rejected, chosen, rejected, ...
    chosen_log_prob, rejected_log_prob = log_prob[::2], log_prob[1::2]
    chosen_mask, rejected_mask = eos_mask[::2], eos_mask[1::2]
    chosen_log_prob_sum = (chosen_log_prob * chosen_mask).sum(-1)
    rejected_log_prob_sum = (rejected_log_prob * rejected_mask).sum(-1)

    if ref_log_prob is None:
        raise NotImplementedError("DPO requires valid ref_log_prob")
    chosen_ref_log_prob, rejected_ref_log_prob = ref_log_prob[::2], ref_log_prob[1::2]
    chosen_ref_log_prob_sum = (chosen_ref_log_prob * chosen_mask).sum(-1)
    rejected_ref_log_prob_sum = (rejected_ref_log_prob * rejected_mask).sum(-1)

    # compute logits
    chosen_ratios = chosen_log_prob_sum - chosen_ref_log_prob_sum
    rejected_ratios = rejected_log_prob_sum - rejected_ref_log_prob_sum
    logits = chosen_ratios - rejected_ratios

    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
        loss = losses.mean()

    else:
        raise NotImplementedError(f"loss_type {loss_type} is not supported in DPO")

    chosen_reward = beta * chosen_ratios.detach()
    rejected_reward = beta * rejected_ratios.detach()
    return loss, chosen_reward, rejected_reward


def compute_policy_loss_pairwise_opmd(
    old_log_prob, log_prob, token_level_scores, eos_mask, index, tau
):
    """Compute policy loss for pairwise_opmd

    NOTE: NOT TESTED YET

    TODO: allow using old_log_prob; for now we just discard it.

    NOTE: use token_level_scores rather than token_level_rewards, because we're not sure yet
    whether this algorithm is compatible with kl penalty as negative reward

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        token_level_scores: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)` or None (when use_uid is False)
        tau: `float`

    Returns:
        opmd_loss: `a scalar torch.Tensor`
            pairwise_opmd loss
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped
        ppo_kl: (float) ... (TODO, confirm that this is only used for logging stats)

    """

    # dummy computation
    log_prob_diff = log_prob - log_prob
    pg_clipfrac = verl_F.masked_mean(torch.gt(log_prob_diff, log_prob_diff).float(), eos_mask)
    ppo_kl = verl_F.masked_mean(-log_prob_diff, eos_mask)

    # loss for pairwise_opmd
    scores = token_level_scores.sum(dim=-1)
    action_level_log_prob = (log_prob * eos_mask).sum(dim=-1)
    diffs = scores - tau * (action_level_log_prob - action_level_log_prob.detach())

    if index is None:
        normalizer = eos_mask.sum() * max(1.0, tau)
        opmd_loss = (diffs - diffs.mean()).square().sum() / normalizer
    else:
        opmd_loss = None
        unique_index = list(set(index.tolist()))
        for idx in unique_index:
            subdiff = diffs[index == idx]
            if subdiff.shape[0] == 1:
                continue
            # subloss = len(subdiff) * subdiff.square().sum() - subdiff.sum().square()
            subloss = (subdiff - subdiff.mean()).square().sum()
            if opmd_loss is None:
                opmd_loss = subloss
            else:
                opmd_loss = opmd_loss + subloss
        normalizer = eos_mask.sum() * max(1.0, tau)
        opmd_loss = opmd_loss / normalizer

    # NOTE: return pg_clipfrac and ppo_kl merely for compatibility with original compute_policy_loss
    return opmd_loss, pg_clipfrac, ppo_kl


def compute_policy_loss_opmd(old_log_prob, log_prob, advantages, eos_mask, tau):
    """The OPMD counterpart of verl's original compute_policy_loss (now renamed as compute_policy_loss_ppo)

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        tau: `float`

    Returns:
        opmd_loss: `a scalar torch.Tensor`
            opmd loss
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped
        ppo_kl: (float) ... (TODO, confirm that this is only used for logging stats)

    """
    log_prob_diff = log_prob - old_log_prob
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(log_prob_diff, log_prob_diff).float(), eos_mask
    )  # meaningless
    ppo_kl = verl_F.masked_mean(-log_prob_diff, eos_mask)

    # --- version 0: kimi-opmd ---

    # # the original quadratic loss in OPMD can be reformulated as follows
    # pg_losses = -advantages * log_prob
    # pg_loss = verl_F.masked_sum(pg_losses, eos_mask)

    # reg_losses = (log_prob_diff * eos_mask).sum(dim=-1).square()
    # reg_loss = reg_losses.sum()

    # opmd_loss = (pg_loss + 0.5 * tau * reg_loss) / eos_mask.sum()
    # # NOTE: this implementation uses batch-wise normalization;
    # # would it be beneficial to use trajectory-wise or group-wise normalization?

    # opmd_loss = opmd_loss / max(1.0, tau)  # for stability when tau is large

    # --- version 1: min-opmd (minimalistic, but theoretically grounded) ---

    pg_losses = -advantages * log_prob
    opmd_loss = verl_F.masked_mean(pg_losses, eos_mask)
    opmd_loss = opmd_loss / (1.0 + tau)  # for regularization (w.r.t. current pi_theta)

    # NOTE: return pg_clipfrac and ppo_kl merely for compatibility with original compute_policy_loss
    return opmd_loss, pg_clipfrac, ppo_kl


def compute_policy_loss_ppo(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_policy_loss_sft(log_prob, eos_mask):
    """Simple way to compute SFT loss, unified with PG loss

    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        sft_loss: `a scalar torch.Tensor`
        pg_clipfrac: dummy value, merely for compatibility
        ppo_kl: dummy value, merely for compatibility

    """
    log_prob_diff = log_prob - log_prob.detach()
    pg_clipfrac = verl_F.masked_mean(torch.gt(log_prob_diff, log_prob_diff).float(), eos_mask)
    ppo_kl = verl_F.masked_mean(-log_prob_diff, eos_mask)

    sft_loss = verl_F.masked_mean(-log_prob, eos_mask)

    # Return pg_clipfrac and ppo_kl merely for compatibility with original compute_policy_loss
    return sft_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
