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

        # Process tensors
        tensors = {
            k: tensor[~is_expert_mask] for k, tensor in exps.batch.items()
        }

        # Process non-tensors
        non_tensors = {
            k: v[~is_expert_mask.detach().cpu().numpy()] for k, v in exps.non_tensor_batch.items()
        }

        # Build new DataProto
        exps = DataProto.from_dict(
            tensors=tensors,
            non_tensors=non_tensors,
            meta_info=exps.meta_info
        )
        return super().__call__(exps, **kwargs)

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 1e-6,
        }
