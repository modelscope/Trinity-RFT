# -*- coding: utf-8 -*-
"""The most simple Importance Sampling policy loss.

loss = -(prob_ratio * advantages).sum()
where prob_ratio = exp(current_logprobs - sampling_logprobs)

Note: This loss is used for on-policy distillation.
"""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


class ImportanceSamplingLossFn(PolicyLossFn):
    """Pure importance sampling loss without clipping.

    loss = -(ratio * advantages)
    where ratio = exp(logprob - old_logprob)
    """

    def __init__(
        self,
        backend: str = "verl",
        loss_agg_mode: str = "token-mean",
    ) -> None:
        super().__init__(backend=backend)
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        # prob_ratio = exp(current_logprobs - sampling_logprobs)
        log_ratio = logprob - old_logprob
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        # loss = -(prob_ratio * advantages)
        pg_losses = -advantages * ratio
        pg_loss = aggregate_loss(pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)

        metrics = {
            "pg_loss": pg_loss.detach().item(),
            "ratio/mean": masked_mean(ratio, action_mask).detach().item(),
            "approx_kl": masked_mean(-log_ratio, action_mask).detach().item(),
        }

        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {"loss_agg_mode": "token-mean"}
