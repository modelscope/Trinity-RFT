"""Mix advantage computation"""

from typing import Dict, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn import ADVANTAGE_FN
from trinity.algorithm.advantage_fn.grpo_advantage import GRPOAdvantageFn


@ADVANTAGE_FN.register_module("mix")
class MIXAdvantageFn(GRPOAdvantageFn):
    """MIX advantage computation"""

    def __init__(
        self,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(epsilon)

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        is_expert_mask = exps.batch["is_expert_mask"]
        device = is_expert_mask.device
        batch_size = is_expert_mask.shape[0]

        # Process tensors
        tensors = {k: tensor[~is_expert_mask] for k, tensor in exps.batch.items()}

        # Process non-tensors
        non_tensors = {
            k: v[~is_expert_mask.detach().cpu().numpy()] for k, v in exps.non_tensor_batch.items()
        }

        # Build new DataProto
        new_exps = DataProto.from_dict(
            tensors=tensors, non_tensors=non_tensors, meta_info=exps.meta_info
        )
        new_exps, new_metrics = super().__call__(new_exps, **kwargs)

        # Get full advantages
        full_advantages = torch.zeros(
            (batch_size, new_exps.batch["advantages"].shape[1]), device=device
        )
        full_returns = torch.zeros((batch_size, new_exps.batch["returns"].shape[1]), device=device)

        # Fill in the non-expert parts with computed values
        full_advantages[~is_expert_mask] = new_exps.batch["advantages"]
        full_returns[~is_expert_mask] = new_exps.batch["returns"]

        # Write back to original exps
        exps.batch["advantages"] = full_advantages
        exps.batch["returns"] = full_returns
        # TODO: change new_metrics
        return exps, new_metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 1e-6,
        }
