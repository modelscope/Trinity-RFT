"""Mix policy loss function."""

from typing import Dict, List, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.policy_loss_fn.ppo_policy_loss import PPOPolicyLossFn
from trinity.algorithm.policy_loss_fn.sft_loss import SFTLossFn


@POLICY_LOSS_FN.register_module("mix")
class MIXPolicyLossFn(PolicyLossFn):
    def __init__(
        self,
        mu: float = 0.1,
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        use_dynamic_bsz: Optional[int] = None,
        ppo_mini_batch_size: Optional[int] = None,
        gradient_accumulation: Optional[int] = None,
        read_batch_size_usual: Optional[int] = None,
        read_batch_size_expert: Optional[int] = None,
        use_token_level_loss_in_sft: bool = False,
    ) -> None:
        self.mu = mu
        self.use_dynamic_bsz = use_dynamic_bsz
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.gradient_accumulation = gradient_accumulation
        self.read_batch_size_usual = read_batch_size_usual
        self.read_batch_size_expert = read_batch_size_expert
        self.grpo_loss_fn = PPOPolicyLossFn(
            clip_range=clip_range,
            clip_range_low=clip_range_low,
            clip_range_high=clip_range_high,
        )
        self.sft_loss_fn = SFTLossFn(use_token_level_loss=use_token_level_loss_in_sft)

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        is_expert_mask = kwargs.get("is_expert_mask", None)
        if is_expert_mask is None:
            raise ValueError("is_expert_mask is required in MIX")
        assert (
            len(is_expert_mask) == logprob.shape[0]
        ), f"Error: {len(is_expert_mask)=} != {logprob.shape[0]=}"

        n_usual_exp = torch.sum(~is_expert_mask).item()
        n_expert_exp = torch.sum(is_expert_mask).item()

        if self.use_dynamic_bsz:
            per_micro_batch_weight_usual = self.ppo_mini_batch_size / (
                logprob.shape[0] * self.read_batch_size_usual
            )
            per_micro_batch_weight_expert = self.ppo_mini_batch_size / (
                logprob.shape[0] * self.read_batch_size_expert
            )
        else:
            per_micro_batch_weight_usual = self.gradient_accumulation / self.read_batch_size_usual  # type: ignore
            per_micro_batch_weight_expert = self.gradient_accumulation / self.read_batch_size_expert  # type: ignore

        if n_usual_exp > 0:
            grpo_loss, grpo_metrics = self.grpo_loss_fn(
                logprob[~is_expert_mask],
                old_logprob[~is_expert_mask],
                action_mask[~is_expert_mask],
                advantages[~is_expert_mask],
                **kwargs,
            )
            grpo_loss = grpo_loss * n_usual_exp * per_micro_batch_weight_usual
            grpo_metrics = {
                k: v * n_usual_exp * per_micro_batch_weight_usual for k, v in grpo_metrics.items()
            }
        else:
            grpo_loss = torch.tensor(0.0, device=logprob.device)
            grpo_metrics = {}

        # SFT Loss (expert)
        if n_expert_exp > 0:
            sft_loss, sft_metrics = self.sft_loss_fn(
                logprob[is_expert_mask],
                action_mask[is_expert_mask],
            )
            sft_loss = sft_loss * n_expert_exp * per_micro_batch_weight_expert
            sft_metrics = {
                k: v * n_expert_exp * per_micro_batch_weight_expert for k, v in sft_metrics.items()
            }
        else:
            sft_loss = torch.tensor(0.0, device=logprob.device)
            sft_metrics = {}

        loss = (1 - self.mu) * grpo_loss + self.mu * sft_loss

        metrics = {f"usual/{k}": v for k, v in grpo_metrics.items()}
        metrics.update({f"expert/{k}": v for k, v in sft_metrics.items()})
        metrics.update({"loss": loss.item()})

        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "mu": 0.1,
            "clip_range": 0.2,
        }

    @property
    def select_keys(self) -> List[str]:
        return ["old_logprob", "action_mask", "advantages", "is_expert_mask"]
