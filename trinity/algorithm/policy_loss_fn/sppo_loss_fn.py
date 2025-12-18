"""sPPO-token policy loss function.
Relevant paper: https://arxiv.org/abs/2108.05828.
"""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


class sPPOPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        epsilon: float = 0.3,
        loss_agg_mode: str = "token-mean",
    ) -> None:
        super().__init__(backend=backend)
        self.epsilon = epsilon
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,  # [batch_size, seq_len]
        old_logprob: torch.Tensor,  # [batch_size, seq_len]
        action_mask: torch.Tensor,  # [batch_size, seq_len]
        advantages: torch.Tensor,  # [batch_size, seq_len]
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Calculate sPPO loss.
        The formula is as follows:
        advantages*log(clip(ratio, 1/(1+epsilon), 1+epsilon))
        ratio = exp(logprob - old_logprob)
        """
        #
        # token-wise
        ratio = torch.exp(logprob - old_logprob).detach()
        is_in_range = (ratio >= (1 / (1 + self.epsilon))) * (ratio <= (1 + self.epsilon))
        is_clipped_mask = ~is_in_range
        pg_losses = -advantages * (logprob - old_logprob) * is_in_range.float()
        pg_loss = aggregate_loss(pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)
        pg_clipfrac = masked_mean(is_clipped_mask.float(), action_mask)
        metrics = {
            "pg_clipfrac": pg_clipfrac.item(),
            "pg_loss": pg_loss.detach().item(),
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 0.3,
            "loss_agg_mode": "token-mean",
        }
