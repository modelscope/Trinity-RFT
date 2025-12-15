"""RLOO advantage computation

Ref: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

from collections import defaultdict
from typing import Dict, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn import AdvantageFn


class RLOOAdvantageFn(AdvantageFn):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        exps: DataProto,
        **kwargs,
    ) -> Tuple[DataProto, Dict]:
        """
        Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

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

        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx], dtype=torch.float32))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for i in range(bsz):
                response_num = len(id2score[index[i]])
                if response_num > 1:
                    scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                        index[i]
                    ] * response_num / (response_num - 1)
            scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        exps.batch["advantages"] = scores
        exps.batch["returns"] = scores

        metrics = {
            # TODO: add meaningful metrics
        }

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {}
