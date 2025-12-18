"""DPO loss function."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import masked_sum


class DPOLossFn(PolicyLossFn):
    def __init__(
        self,
        backend: str = "verl",
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(backend=backend)
        self.beta = beta
        self.label_smoothing = label_smoothing

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        ref_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        chosen_logprob = logprob[::2]
        rejected_logprob = logprob[1::2]
        chosen_mask = action_mask[::2]
        rejected_mask = action_mask[1::2]
        chosen_logprob_sum = masked_sum(chosen_logprob, chosen_mask)
        rejected_logprob_sum = masked_sum(rejected_logprob, rejected_mask)

        chosen_ref_logprob = ref_logprob[::2]
        rejected_ref_logprob = ref_logprob[1::2]
        chosen_ref_logprob_sum = masked_sum(chosen_ref_logprob, chosen_mask)
        rejected_ref_logprob_sum = masked_sum(rejected_ref_logprob, rejected_mask)

        chosen_ratios = chosen_logprob_sum - chosen_ref_logprob_sum
        rejected_ratios = rejected_logprob_sum - rejected_ref_logprob_sum
        logits = chosen_ratios - rejected_ratios
        # TODO: support other loss functions
        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )
        loss = losses.mean()
        chosen_reward = self.beta * chosen_ratios.detach().mean().item()
        rejected_reward = self.beta * rejected_ratios.detach().mean().item()
        accuracy_mean = (chosen_ratios.detach() > rejected_ratios.detach()).float().mean().item()
        return loss, {
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "accuracy_mean": accuracy_mean,
            "dpo_loss": loss.detach().item(),
        }

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "beta": 0.1,
            "label_smoothing": 0.0,
        }
