"""AsymRE advantage computation"""

from collections import defaultdict
from typing import Dict, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn import ADVANTAGE_FN, AdvantageFn


@ADVANTAGE_FN.register_module("asymre")
class ASYMREAdvantageFn(AdvantageFn):
    """AsymRE advantage computation"""

    def __init__(
        self,
        asymre_baseline_shift: float = -0.1,
    ) -> None:
        self.asymre_baseline_shift = asymre_baseline_shift

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        """Modified from compute_grpo_outcome_advantage

        Compute advantage for AsymRE, operating only on Outcome reward
        (with only one scalar reward for each response).

            token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
            eos_mask: `(torch.Tensor)`
                shape: (bs, response_length)
            scores: `(torch.Tensor)`
                shape: (bs, response_length)
        """
        token_level_rewards = exps.batch["token_level_rewards"]
        eos_mask = exps.batch["response_mask"]
        
        index = exps.non_tensor_batch["uid"]
        asymre_baseline_shift = self.asymre_baseline_shift

        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2baseline = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2baseline[idx] = torch.tensor(0.0)
                    # TODO: consider id2baseline[idx] = id2score[idx] (so that this sample won't take effect?)
                elif len(id2score[idx]) > 1:
                    id2baseline[idx] = torch.mean(torch.tensor(id2score[idx])) + asymre_baseline_shift
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                scores[i] = scores[i] - id2baseline[index[i]]
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        exps.batch["advantages"] = scores
        exps.batch["returns"] = scores

        metrics = {
            # TODO: add meaningful metrics
        }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "asymre_baseline_shift": -0.1,
        }
