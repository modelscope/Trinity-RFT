"""SFT loss function."""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_loss


@POLICY_LOSS_FN.register_module("sft")
class SFTLossFn(PolicyLossFn):
    def __init__(self, backend: str = "verl", use_token_level_loss: bool = True) -> None:
        super().__init__(backend=backend)
        self.use_token_level_loss = use_token_level_loss

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        if self.use_token_level_loss:
            sft_loss = masked_loss(-logprob, action_mask, loss_agg_mode="token-mean")
        else:
            sft_loss = masked_loss(-logprob, action_mask, loss_agg_mode="seq-mean-token-mean")

        return sft_loss, {"sft_loss": sft_loss.detach().item()}

    @classmethod
    def default_args(cls):
        return {
            "use_token_level_loss": True,
        }
