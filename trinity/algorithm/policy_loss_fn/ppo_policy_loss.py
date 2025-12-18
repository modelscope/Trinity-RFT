"""PPO policy loss function.

Modified from https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


class PPOPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        clip_ratio_c: float = 3.0,
        loss_agg_mode: Optional[str] = "token-mean",
        enable_sequence_masking: bool = False,  # introduced in DeepseekV3.2
        delta_sequence_masking: float = 0.1,
        fallback_to_policy_gradient: bool = False,
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
        self.enable_sequence_masking = enable_sequence_masking
        self.delta_sequence_masking = delta_sequence_masking
        self.fallback_to_policy_gradient = fallback_to_policy_gradient

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        negative_approx_kl = logprob - old_logprob
        if self.fallback_to_policy_gradient:
            # ignore vllm logprob difference and use pure policy gradient loss
            negative_approx_kl = logprob - logprob.detach()
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

        # Apply sequence masking if enabled
        if self.enable_sequence_masking:
            # Compute sequence-level KL divergence: mean KL per sequence
            # Shape: (batch_size, seq_len) -> (batch_size,)
            kl_per_token = -negative_approx_kl  # KL divergence per token
            sequence_kl = (kl_per_token * action_mask).sum(dim=-1) / (
                action_mask.sum(dim=-1) + 1e-10
            )

            # Create mask: mask out tokens with negative advantages when sequence KL is high
            # Token-level advantage check: (batch_size, seq_len)
            has_negative_advantage = advantages < 0
            # Sequence-level KL check: (batch_size,) -> (batch_size, 1) -> (batch_size, seq_len)
            exceeds_kl_threshold = (
                (sequence_kl > self.delta_sequence_masking).unsqueeze(-1).expand_as(advantages)
            )
            # Mask tokens that are both negative advantage AND in high-KL sequences
            should_mask = has_negative_advantage & exceeds_kl_threshold
            sequence_mask = (~should_mask).float()

            # Apply sequence mask to the losses
            pg_losses = pg_losses * sequence_mask

            metrics_seq_mask = {
                "seq_mask/masked_tokens": should_mask.float().sum().item()
                / (action_mask.sum().item() + 1e-10),
                "seq_mask/mean_sequence_kl": sequence_kl.mean().detach().item(),
            }

        pg_loss = aggregate_loss(pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)
        metrics = {
            "pg_clipfrac": pg_clip_frac.detach().item(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
        }
        if self.enable_sequence_masking:
            metrics.update(metrics_seq_mask)
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "clip_range": 0.2,
            "clip_ratio_c": 3.0,
            "loss_agg_mode": "token-mean",
            "enable_sequence_masking": False,
            "delta_sequence_masking": 0.1,
            "fallback_to_policy_gradient": False,
        }
