"""PPO policy loss function.

Modified from https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_loss, masked_mean


@POLICY_LOSS_FN.register_module("ppo")
class PPOPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        loss_agg_mode: Optional[str] = "token-mean",
        truncate_adv_pos_is: bool = False,
        truncate_adv_neg_is: bool = False,
        truncate_is_range_low: Optional[float] = 0.0,
        truncate_is_range_high: Optional[float] = 2.0,
    ) -> None:
        """
        Initialize PPO policy loss function.

        Args:
            backend: Backend framework (default: "verl")
            clip_range: Symmetric clipping range for PPO
            clip_range_low: Lower bound for clipping (1.0 - clip_range_low)
            clip_range_high: Upper bound for clipping (1.0 + clip_range_high)
            loss_agg_mode: Loss aggregation mode (default: "token-mean")
            truncate_adv_pos_is: Whether to truncate large importance sampling ratios
                when advantage is positive to handle calculation discrepancies between
                rollout and training engines
            truncate_adv_neg_is: Whether to truncate large importance sampling ratios
                when advantage is negative to handle calculation discrepancies between
                rollout and training engines
            truncate_is_range_low: Lower bound for IS ratio truncation (default: 0.0)
            truncate_is_range_high: Upper bound for IS ratio truncation (default: 2.0)
        """
        super().__init__(backend=backend)
        if clip_range_low is None:
            self.clip_range_low = clip_range
        else:
            self.clip_range_low = clip_range_low
        if clip_range_high is None:
            self.clip_range_high = clip_range
        else:
            self.clip_range_high = clip_range_high
        assert self.clip_range_low is not None, "clip_range_low must be specified."
        assert self.clip_range_high is not None, "clip_range_high must be specified."
        self.loss_agg_mode = loss_agg_mode

        # Truncate large IS configuration
        self.truncate_adv_pos_is = truncate_adv_pos_is
        self.truncate_adv_neg_is = truncate_adv_neg_is
        if truncate_adv_pos_is:
            self.truncate_is_range_low = truncate_is_range_low
            assert (
                self.truncate_is_range_low is not None
            ), "truncate_is_range_low must be specified."
            assert (
                self.truncate_is_range_low >= 0.0
            ), "truncate_is_range_low must be non-negative."
            assert (self.truncate_is_range_low < 1.0-self.clip_range_low
            ), "truncate_is_range_low must be less than 1.0 - clip_range_low."
        if truncate_adv_neg_is:
            self.truncate_is_range_high = truncate_is_range_high
            assert (
                self.truncate_is_range_high is not None
            ), "truncate_is_range_high must be specified."
            assert (
                self.truncate_is_range_high > 1.0+self.clip_range_high
            ), "truncate_is_range_high must be greater than clip_range_high + 1.0."
        if truncate_adv_pos_is and truncate_adv_neg_is:
            assert (
                self.truncate_is_range_high > self.truncate_is_range_low
            ), "truncate_is_range_high must be greater than truncate_is_range_low."

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        negative_approx_kl = logprob - old_logprob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = masked_mean(-negative_approx_kl, action_mask)

        # First clipping by clip_range, and calculate pg_clipfrac
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_range_low, 1.0 + self.clip_range_high  # type: ignore
        )
        pg_losses_clip = torch.maximum(pg_losses1, pg_losses2)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)

        # After clipped by clip_range, further truncate IS ratios if enabled
        # This helps stabilize training when there are calculation discrepancies between
        # rollout and training engines, especially for small probabilities
        pg_truncfrac_pos, pg_truncfrac_neg = 0.0, 0.0
        pg_losses_trunc = pg_losses_clip
        
        # Add IS truncation for positive advantages
        if self.truncate_adv_pos_is:
            pg_losses_pos_trunc = -advantages * self.truncate_is_range_low
            pg_truncfrac_pos = masked_mean(
                torch.lt(pg_losses_pos_trunc, pg_losses_trunc) * (advantages > 0).float(),
                action_mask,
            )
            pg_losses_pos = torch.minimum(pg_losses_trunc, pg_losses_pos_trunc)
            pg_losses_trunc = torch.where(advantages > 0, pg_losses_pos, pg_losses_trunc)

        # Add IS truncation for negative advantages
        if self.truncate_adv_neg_is:
            pg_losses_neg_trunc = -advantages * self.truncate_is_range_high
            pg_truncfrac_neg = masked_mean(
                torch.lt(pg_losses_neg_trunc, pg_losses_trunc) * (advantages < 0).float(),
                action_mask,
            )
            pg_losses_neg = torch.minimum(pg_losses_trunc, pg_losses_neg_trunc)
            pg_losses_trunc = torch.where(advantages < 0, pg_losses_neg, pg_losses_trunc)

        pg_loss = masked_loss(
            pg_losses_trunc, action_mask, loss_agg_mode=self.loss_agg_mode
        )
        metrics = {
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
        }

        # Add IS truncation metrics if enabled
        if self.truncate_adv_pos_is:
            metrics["is_truncate_frac_pos"] = pg_truncfrac_pos.detach().item()
        if self.truncate_adv_neg_is:
            metrics["is_truncate_frac_neg"] = pg_truncfrac_neg.detach().item()

        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "clip_range": 0.2,
            "loss_agg_mode": "token-mean",
            "truncate_adv_pos_is": False,
            "truncate_adv_neg_is": False,
            "truncate_is_range_low": 0.0,
            "truncate_is_range_high": 2.0,
        }
