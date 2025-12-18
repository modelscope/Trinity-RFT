"""Mix policy loss function."""

from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.policy_loss_fn.ppo_policy_loss import PPOPolicyLossFn
from trinity.algorithm.policy_loss_fn.sft_loss import SFTLossFn


class MIXPolicyLossFn(PolicyLossFn):
    """Implements a mixed policy loss combining GRPO and SFT losses.

    This loss function applies different loss components to data based on whether
    it comes from an expert or not, as indicated by `expert_mask`. It combines:
    - GRPO loss (self.grpo_loss_fn) for non-expert data
    - SFT loss (self.sft_loss_fn) for expert data
    - Weighting parameter `mu`

    The per-sample weights are normalized using either `experience_per_gpu` or
    `gradient_accumulation`, depending on whether dynamic batch sizing is enabled,
    to ensure consistent weighting across different batches of the same type experiences.
    """

    def __init__(
        self,
        backend: str = "verl",
        mu: float = 0.1,
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        use_dynamic_bsz: Optional[bool] = None,
        ppo_mini_batch_size: int = 1,
        ppo_micro_batch_size_per_gpu: int = 1,
        ngpus_trainer: int = 1,
        train_batch_size_usual: int = 1,
        train_batch_size_expert: int = 1,
        loss_agg_mode: str = "token-mean",
        sft_loss_agg_mode: Optional[str] = None,
        grpo_loss_agg_mode: Optional[str] = None,
    ) -> None:
        super().__init__(backend=backend)
        self.mu = mu
        self.use_dynamic_bsz = use_dynamic_bsz
        self.experience_per_gpu = ppo_mini_batch_size // ngpus_trainer
        self.gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu
        self.train_batch_size_usual = train_batch_size_usual // ngpus_trainer
        self.train_batch_size_expert = train_batch_size_expert // ngpus_trainer
        self.grpo_loss_fn = PPOPolicyLossFn(
            clip_range=clip_range,
            clip_range_low=clip_range_low,
            clip_range_high=clip_range_high,
            loss_agg_mode=grpo_loss_agg_mode or loss_agg_mode,
        )
        self.sft_loss_fn = SFTLossFn(loss_agg_mode=sft_loss_agg_mode or loss_agg_mode)

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        expert_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        assert (
            len(expert_mask) == logprob.shape[0]
        ), f"Error: {len(expert_mask)=} != {logprob.shape[0]=}"

        n_usual_exp = torch.sum(~expert_mask).item()
        n_expert_exp = torch.sum(expert_mask).item()

        if self.use_dynamic_bsz:
            per_micro_batch_weight_usual = self.experience_per_gpu / (
                logprob.shape[0] * self.train_batch_size_usual
            )
            per_micro_batch_weight_expert = self.experience_per_gpu / (
                logprob.shape[0] * self.train_batch_size_expert
            )
        else:
            per_micro_batch_weight_usual = self.gradient_accumulation / self.train_batch_size_usual  # type: ignore
            per_micro_batch_weight_expert = self.gradient_accumulation / self.train_batch_size_expert  # type: ignore

        if n_usual_exp > 0:
            grpo_loss, grpo_metrics = self.grpo_loss_fn(
                logprob[~expert_mask],
                old_logprob[~expert_mask],
                action_mask[~expert_mask],
                advantages[~expert_mask],
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
                logprob[expert_mask],
                action_mask[expert_mask],
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
        metrics["loss"] = loss.item()

        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "mu": 0.1,
            "clip_range": 0.2,
            "loss_agg_mode": "token-mean",
        }
