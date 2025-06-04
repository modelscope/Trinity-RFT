"""PPO policy loss function.

Modified from https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from typing import Dict, List, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_mean


@POLICY_LOSS_FN.register_module("opmd")
class OPMDPolicyLossFn(PolicyLossFn):
    def __init__(self, tau: float = 1.0) -> None:
        self.tau = tau

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,  # NOT USED!
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        pg_losses = -advantages * logprob
        opmd_loss = masked_mean(pg_losses, action_mask)
        opmd_loss = opmd_loss / (1.0 + self.tau)  # for regularization (w.r.t. current pi_theta)
        return opmd_loss, {"opmd_loss": opmd_loss.detach().item()}

    @classmethod
    def default_args(cls) -> Dict:
        return {"tau": 1.0}

    @property
    def select_keys(self) -> List[str]:
        return [
            "old_logprob",
            "action_mask",
            "advantages",
        ]
