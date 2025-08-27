"""ReCLIP-token policy loss function.

Relevant paper: https://arxiv.org/abs/2108.05828.
"""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_mean


@POLICY_LOSS_FN.register_module("reclip")
class ReCLIPPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(backend=backend)
        self.epsilon = epsilon

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,  # [batch_size, seq_len]
        old_logprob: torch.Tensor,  # [batch_size, seq_len]
        action_mask: torch.Tensor,  # [batch_size, seq_len]
        advantages: torch.Tensor,  # [batch_size, seq_len]
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        # token-wise
        negative_approx_kl = logprob - old_logprob
        ratio = torch.clamp(torch.exp(negative_approx_kl), 1 / (1 + self.epsilon), 1 + self.epsilon)
        pg_losses_clipped = -advantages * ratio
        pg_losses_unclipped = -advantages * torch.exp(negative_approx_kl)
        pg_loss = masked_mean(pg_losses_clipped, action_mask)
        pg_clipfrac = masked_mean(
            torch.gt(pg_losses_clipped, pg_losses_unclipped).float(), action_mask
        )
        metrics = {
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "pg_loss": pg_loss.detach().item(),
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 0.3,
        }
