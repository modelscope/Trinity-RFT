"""SFT loss function."""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss


class SFTLossFn(PolicyLossFn):
    def __init__(self, backend: str = "verl", loss_agg_mode: str = "token-mean") -> None:
        super().__init__(backend=backend)
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        sft_loss = aggregate_loss(-logprob, action_mask, loss_agg_mode=self.loss_agg_mode)

        return sft_loss, {"sft_loss": sft_loss.detach().item()}

    @classmethod
    def default_args(cls):
        return {
            "loss_agg_mode": "token-mean",
        }
