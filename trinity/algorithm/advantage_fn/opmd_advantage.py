"""OPMD advantage computation"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn, GroupAdvantage
from trinity.common.experience import Experience, group_by
from trinity.utils.annotations import Deprecated


@Deprecated
class OPMDAdvantageFn(AdvantageFn):
    """OPMD advantage computation"""

    def __init__(
        self,
        opmd_baseline: str = "mean",
        tau: float = 1.0,
    ) -> None:
        self.opmd_baseline = opmd_baseline
        self.tau = tau

    def __call__(
        self,
        exps: "DataProto",
        **kwargs,
    ) -> Tuple["DataProto", Dict]:
        """Modified from compute_grpo_outcome_advantage

        Compute advantage for OPMD, operating only on Outcome reward
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
        # TODO (yanxi): confirm consistency with exps.batch["attention_mask"][:, -response_length:] in original implementation
        index = exps.non_tensor_batch["uid"]
        opmd_baseline = self.opmd_baseline
        tau = self.tau

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
                    if opmd_baseline == "mean":
                        id2baseline[idx] = torch.mean(
                            torch.tensor(id2score[idx], dtype=torch.float32)
                        )
                    elif opmd_baseline == "logavgexp":
                        rewards_tensor = torch.tensor(id2score[idx], dtype=torch.float32)
                        # here we use the fact that logavgexp(x) = logsumexp(x) - log(len(x))
                        id2baseline[idx] = tau * (
                            torch.logsumexp(rewards_tensor / tau, dim=-1)
                            - torch.log(torch.tensor(len(id2score[idx])))
                        )
                    else:
                        raise NotImplementedError
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
            "opmd_baseline": "mean",
            "tau": 1.0,
        }


class OPMDGroupAdvantage(GroupAdvantage):
    """OPMD Group Advantage computation"""

    def __init__(self, opmd_baseline: str = "mean", tau: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.opmd_baseline = opmd_baseline
        self.tau = tau

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            group_rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
            reward_mean = torch.mean(group_rewards)
            if len(exps) == 1:
                group_baseline = torch.tensor(0.0)
            else:
                if self.opmd_baseline == "mean":
                    group_baseline = reward_mean
                else:
                    group_baseline = self.tau * (
                        torch.logsumexp(group_rewards / self.tau, dim=-1)
                        - torch.log(torch.tensor(len(exps)))
                    )
            for exp in exps:
                score = exp.reward - group_baseline
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()
            metrics = {
                "group_baseline": group_baseline.item(),
                "reward_mean": reward_mean.item(),
            }
        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"opmd_baseline": "mean", "tau": 1.0}
