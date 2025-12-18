"""OPMD policy loss function."""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss


class OPMDPolicyLossFn(PolicyLossFn):
    def __init__(
        self, backend: str = "verl", tau: float = 1.0, loss_agg_mode: str = "token-mean"
    ) -> None:
        super().__init__(backend=backend)
        self.tau = tau
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        pg_losses = -advantages * logprob
        opmd_loss = aggregate_loss(pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)
        opmd_loss = opmd_loss / (1.0 + self.tau)  # for regularization (w.r.t. current pi_theta)
        return opmd_loss, {"opmd_loss": opmd_loss.detach().item()}

    @classmethod
    def default_args(cls) -> Dict:
        return {"tau": 1.0, "loss_agg_mode": "token-mean"}
