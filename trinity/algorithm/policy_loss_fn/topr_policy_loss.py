"""TOPR policy loss function.
Refer to https://arxiv.org/pdf/2503.14286v1
"""
from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


class TOPRPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        advantage_threshold: float = 0.0,
        loss_agg_mode: str = "token-mean",
    ) -> None:
        super().__init__(backend=backend)
        self.advantage_threshold = advantage_threshold
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,  # In TOPR, this is actually the rewards R(x,y)
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute TOPR policy loss.

        In TOPR:
        - α = [π(y|x)/μ(y|x)]_0^1 if R(x,y) <= threshold else 1
        - loss = -sg(α) * r(x,y) * log π(y|x)
        """
        # in Orginal TOPR paper, advantages are simply rewards
        # However, we can use advantages as rewards(Baseline Trick)
        rewards = advantages

        # Compute ratio π(y|x) / μ(y|x) in log space for numerical stability
        log_ratio = logprob - old_logprob
        ratio = torch.exp(log_ratio)
        ratio_clamped = torch.clamp(ratio, min=0.0, max=1.0)

        # Apply TOPR's conditional weighting:
        # α = ratio clamp min=0 max=1 if R(x,y) <= threshold else 1
        alpha = torch.where(
            rewards <= self.advantage_threshold, ratio_clamped, torch.ones_like(ratio)
        )

        # TOPR loss: l = -α * r(x,y) * log π(y|x)
        # We want to maximize α * r(x,y) * log π(y|x), so minimize the negative
        topr_loss = -alpha.detach() * rewards * logprob  # detach alpha as it's used with stop-grad

        # Apply masking and compute mean
        loss = aggregate_loss(topr_loss, action_mask, loss_agg_mode=self.loss_agg_mode)

        # Average alpha value for monitoring
        avg_alpha = masked_mean(alpha, action_mask)

        metrics = {
            "topr_loss": loss.detach().item(),
            "avg_alpha": avg_alpha.detach().item(),
            "avg_ratio": masked_mean(ratio, action_mask).detach().item(),
        }

        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "advantage_threshold": 0.0,
            "loss_agg_mode": "token-mean",
        }
