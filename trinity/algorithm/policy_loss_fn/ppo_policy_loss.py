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
        truncate_large_is: bool = False,
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
            truncate_large_is: Whether to truncate large importance sampling ratios
                to handle calculation discrepancies between rollout and training engines
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
        self.truncate_large_is = truncate_large_is
        if truncate_large_is:
            self.truncate_is_range_low = truncate_is_range_low
            self.truncate_is_range_high = truncate_is_range_high
            assert (
                self.truncate_is_range_low is not None
            ), "truncate_is_range_low must be specified."
            assert (
                self.truncate_is_range_high is not None
            ), "truncate_is_range_high must be specified."
            assert self.truncate_is_range_low >= 0.0, "truncate_is_range_low must be non-negative."
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

        # Truncate large IS ratios if enabled
        # This helps stabilize training when there are calculation discrepancies between
        # rollout and training engines, especially for small probabilities
        if self.truncate_large_is:
            # Track how often truncation occurs (before actually truncating)
            # More efficient than cloning: directly check which values fall outside bounds
            ratio_detached = ratio.detach()
            is_truncate_frac = masked_mean(
                (ratio_detached < self.truncate_is_range_low).float(), action_mask
            ) + masked_mean((ratio_detached > self.truncate_is_range_high).float(), action_mask)
            ratio = torch.clamp(ratio, self.truncate_is_range_low, self.truncate_is_range_high)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_range_low, 1.0 + self.clip_range_high  # type: ignore
        )

        pg_loss = masked_loss(
            torch.max(pg_losses, pg_losses2), action_mask, loss_agg_mode=self.loss_agg_mode
        )
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), action_mask)
        metrics = {
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
        }

        # Add IS truncation metrics if enabled
        if self.truncate_large_is:
            metrics["is_truncate_frac"] = is_truncate_frac.detach().item()

        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "clip_range": 0.2,
            "loss_agg_mode": "token-mean",
            "truncate_large_is": False,
            "truncate_is_range_low": 0.0,
            "truncate_is_range_high": 2.0,
        }
