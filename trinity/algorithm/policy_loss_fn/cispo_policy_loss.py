"""CISPO policy loss function.
Refer to https://arxiv.org/abs/2506.13585 for details.
"""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


class CISPOPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        clip_range_low: float = 1.0,
        clip_range_high: float = 0.28,
        enable_mask_clip: bool = False,
        mask_clip_range_low: float = 1.0,
        mask_clip_range_high: float = 0.28,
        loss_agg_mode: str = "token-mean",
    ) -> None:
        super().__init__(backend=backend)
        self.clip_range_low = clip_range_low
        self.clip_range_high = clip_range_high
        self.enable_mask_clip = enable_mask_clip
        self.mask_clip_range_low = mask_clip_range_low
        self.mask_clip_range_high = mask_clip_range_high
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
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = masked_mean(-negative_approx_kl, action_mask)
        ratio_clamped = torch.clamp(
            ratio, min=1.0 - self.clip_range_low, max=1.0 + self.clip_range_high
        )

        # mask = 0 if ratio > 1.0 + self.clip_range_high and advantages > 0
        # mask = 0 if ratio < 1.0 - self.clip_range_low and advantages < 0
        # else 1
        mask = torch.ones_like(ratio)
        if self.enable_mask_clip:
            mask = torch.where(
                (ratio > 1.0 + self.mask_clip_range_high) & (advantages > 0),
                torch.zeros_like(ratio),
                mask,
            )
            mask = torch.where(
                (ratio < 1.0 - self.mask_clip_range_low) & (advantages < 0),
                torch.zeros_like(ratio),
                mask,
            )

        cispo_loss = -advantages * ratio_clamped.detach() * mask.detach() * logprob

        loss = aggregate_loss(cispo_loss, action_mask, loss_agg_mode=self.loss_agg_mode)
        unmasked_frac = masked_mean(mask, action_mask)

        metrics = {
            "cispo_loss": loss.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "unmasked_frac": unmasked_frac.detach().item(),
        }

        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        """
        In the original paper:
            we did not impose a lower bound on the IS weight by setting clip_range_low to a high value, instead, we only tuned clip_range_high

        """
        return {
            "clip_range_low": 1.0,
            "clip_range_high": 0.28,
            "enable_mask_clip": False,
            "mask_clip_range_low": 1.0,
            "mask_clip_range_high": 0.28,
            "loss_agg_mode": "token-mean",
        }
