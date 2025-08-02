"""GSPO-token policy loss function.

Implemented from https://arxiv.org/pdf/2507.18071
"""

from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_mean


@POLICY_LOSS_FN.register_module("gspo")
class GSPOLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
    ) -> None:
        super().__init__(backend=backend)
        _clip_range_low = clip_range_low if clip_range_low is not None else clip_range
        if _clip_range_low is None:
            raise ValueError("Either clip_range or clip_range_low must be specified.")
        self.clip_range_low = _clip_range_low

        _clip_range_high = clip_range_high if clip_range_high is not None else clip_range
        if _clip_range_high is None:
            raise ValueError("Either clip_range or clip_range_high must be specified.")
        self.clip_range_high = _clip_range_high

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,  # [batch_size, seq_len]
        old_logprob: torch.Tensor,  # [batch_size, seq_len]
        action_mask: torch.Tensor,  # [batch_size, seq_len]
        advantages: torch.Tensor,  # [batch_size, seq_len]
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        negative_approx_kl = logprob - old_logprob  # [batch_size, seq_len]
        negative_approx_kl_seq = masked_mean(
            negative_approx_kl, action_mask, axis=-1
        )  # [batch_size]
        log_seq_importance_ratio = (
            logprob - logprob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
        )  # [batch_size, seq_len]
        ratio = torch.exp(log_seq_importance_ratio)  # [batch_size, seq_len]
        pg_losses = -advantages * ratio  # [batch_size, seq_len]
        pg_losses_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_range_low, 1.0 + self.clip_range_high
        )  # [batch_size, seq_len]

        seq_losses = masked_mean(
            torch.max(pg_losses, pg_losses_clipped), action_mask, axis=-1
        )  # [batch_size]
        pg_loss = torch.mean(seq_losses)
        pg_clipfrac = masked_mean(torch.gt(pg_losses_clipped, pg_losses).float(), action_mask)
        ppo_kl = masked_mean(-negative_approx_kl, action_mask)
        ppo_kl_seq = torch.mean(-negative_approx_kl_seq)
        metrics = {
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
            "ppo_kl_seq": ppo_kl_seq.detach().item(),
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        # See discussion in https://github.com/volcengine/verl/pull/2775#issuecomment-3130065984
        return {
            "clip_range_low": 0.0003,
            "clip_range_high": 0.0004,
        }
