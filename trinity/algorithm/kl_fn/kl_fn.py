"""KL penalty and loss.

Ref:
https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py
https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/utils.py
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from trinity.algorithm.utils import aggregate_loss, masked_mean
from trinity.utils.registry import Registry

KL_FN = Registry("kl_fn")


class KLFn(ABC):
    """
    KL penalty and loss.
    """

    def __init__(
        self,
        adaptive: bool = False,
        kl_coef: float = 0.001,
        target_kl: Optional[float] = None,
        horizon: Optional[float] = None,
    ) -> None:
        self.kl_coef = kl_coef
        self.adaptive = adaptive
        self.target_kl = target_kl
        self.horizon = horizon
        if adaptive and (target_kl is None or horizon is None):
            raise ValueError("Target KL and horizon must be provided for adaptive KL.")

    def update_kl_coef(self, current_kl: float, batch_size: int) -> None:
        """Update kl coefficient."""
        if self.adaptive:
            target_kl = self.target_kl
            proportional_error = torch.clip(current_kl / target_kl - 1, -0.2, 0.2).item()  # type: ignore
            multiplier = 1 + proportional_error * batch_size / self.horizon
            self.kl_coef *= multiplier

    def apply_kl_penalty_to_reward(self, experiences: Any) -> Tuple[Any, Dict]:
        """Apply KL penalty to reward. Only support DataProto input for now."""
        responses = experiences.batch["responses"]
        response_length = responses.size(1)
        token_level_scores = experiences.batch["token_level_scores"]
        batch_size = experiences.batch.batch_size[0]
        attention_mask = experiences.batch["attention_mask"]
        response_mask = experiences.batch["response_mask"]
        assert response_mask.shape == attention_mask[:, -response_length:].shape
        logprob = experiences.batch["old_log_probs"]
        ref_logprob = experiences.batch["ref_log_prob"]

        if "ref_log_prob" in experiences.batch.keys():
            kl = self.calculate_kl(logprob, ref_logprob)
            kl = kl * response_mask
            kl_coef = self.kl_coef
            experiences.batch["token_level_rewards"] = token_level_scores - kl_coef * kl
        else:
            kl_coef = 0.0
            kl = torch.zeros_like(response_mask, dtype=torch.float32)
            experiences.batch["token_level_rewards"] = token_level_scores

        current_kl = masked_mean(kl, mask=response_mask, axis=-1).mean(dim=0).item()
        self.update_kl_coef(current_kl=current_kl, batch_size=batch_size)

        metrics = {
            "kl": current_kl,
            "kl_coef": kl_coef,
        }

        return experiences, metrics

    def calculate_kl_loss(
        self,
        logprob: torch.Tensor,
        ref_logprob: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute KL loss."""
        kl = self.calculate_kl(logprob, ref_logprob)
        kl_loss = aggregate_loss(kl, response_mask, loss_agg_mode=loss_agg_mode)
        metrics = {
            "kl_loss": kl_loss.detach().item(),
            "kl_coef": self.kl_coef,
        }
        return kl_loss * self.kl_coef, metrics

    @abstractmethod
    def calculate_kl(self, logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between logprob and ref_logprob."""

    @classmethod
    def default_args(cls):
        """Get the default initialization arguments."""
        return {"adaptive": False, "kl_coef": 0.001}


@KL_FN.register_module("none")
class DummyKLFn(KLFn):
    """
    Dummy KL function.
    """

    def calculate_kl(self, logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(logprob)

    def apply_kl_penalty_to_reward(self, experiences: Any) -> Tuple[Any, Dict]:
        experiences.batch["token_level_rewards"] = experiences.batch["token_level_scores"]
        return experiences, {}

    def calculate_kl_loss(
        self,
        logprob: torch.Tensor,
        ref_logprob: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str,
    ) -> Tuple[torch.Tensor, Dict]:
        # return a zero tensor
        return torch.tensor(0.0), {}


@KL_FN.register_module("k1")
class K1Fn(KLFn):
    """
    KL K1 function.
    """

    def calculate_kl(self, logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
        return logprob - ref_logprob


@KL_FN.register_module("k2")
class K2Fn(KLFn):
    """
    KL K2 function.
    """

    def calculate_kl(self, logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
        return (logprob - ref_logprob).square() * 0.5


@KL_FN.register_module("k3")
class K3Fn(KLFn):
    """
    KL K3 function.
    """

    def calculate_kl(self, logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
        logr = ref_logprob - logprob
        return logr.exp() - 1 - logr


@KL_FN.register_module("low_var_kl")
class LowVarKLFn(KLFn):
    """
    Low Variance KL function.
    """

    def calculate_kl(self, logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)


@KL_FN.register_module("abs")
class AbsFn(KLFn):
    """
    KL Abs function.
    """

    def calculate_kl(self, logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
        return torch.abs(logprob - ref_logprob)
