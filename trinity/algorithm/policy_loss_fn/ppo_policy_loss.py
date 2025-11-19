"""PPO policy loss function.

Modified from https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


@POLICY_LOSS_FN.register_module("ppo")
class PPOPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        clip_ratio_c: float = 3.0,
        loss_agg_mode: Optional[str] = "token-mean",
    ) -> None:
        super().__init__(backend=backend)
        if clip_range_low is None:
            self.clip_range_low = clip_range
        else:
            self.clip_range_low = clip_range_low
        if clip_range_high is None:
            self.clip_range_high = clip_range
        else:
            self.clip_range_high = clip_range_high
        self.clip_ratio_c = clip_ratio_c
        assert clip_ratio_c > 1.0, "clip_ratio_c must be greater than 1.0."
        assert self.clip_range_low is not None, "clip_range_low must be specified."
        assert self.clip_range_high is not None, "clip_range_high must be specified."
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        negative_approx_kl = logprob - old_logprob
        # Clamp negative_approx_kl for stability
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = masked_mean(-negative_approx_kl, action_mask)

        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_range_low, 1.0 + self.clip_range_high  # type: ignore
        )

        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

        pg_clip_frac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)

        pg_losses3 = -advantages * self.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), action_mask
        )
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        pg_loss = aggregate_loss(pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)
        metrics = {
            "pg_clipfrac": pg_clip_frac.detach().item(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "clip_range": 0.2,
            "clip_ratio_c": 3.0,
            "loss_agg_mode": "token-mean",
        }
