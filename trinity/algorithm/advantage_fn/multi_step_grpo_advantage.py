"""GRPO advantage computation for multi-step scenarios
"""
from typing import Dict, List, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.monitor import gather_metrics


@ADVANTAGE_FN.register_module("step_wise_grpo")
class StepWiseGRPOAdvantageFn(AdvantageFn, ExperienceOperator):
    """
    An advantage function that broadcasts advantages from the last step to previous steps.
    Inspired by rLLM (https://github.com/rllm-org/rllm).
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        enable_step_norm: bool = False,
        **kwargs,
    ) -> None:
        self.epsilon = epsilon
        self.enable_step_norm = enable_step_norm

    def calculate_last_step_advantage(
        self, exps: Dict[str, Experience]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate group advantage for a given group of experiences.

        Args:
            exps (Dict[str, Experience]): One experience per run, keyed by run ID.

        Returns:
            Dict[str, float]: A tuple containing the scores for each run.
            Dict[str, float]: Metrics for logging.
        """
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps.values()], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)
            scores = {}
            for rid, exp in exps.items():
                score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                scores[rid] = score.item()
            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }
        return scores, metrics

    def broadcast_advantages(
        self, run_exps: Dict[str, List[Experience]], scores: Dict[str, float]
    ) -> Dict[str, List[Experience]]:
        """Broadcast the calculated advantages to all previous steps in each run.

        Args:
            run_exps (Dict[str, List[Experience]]): Experiences grouped by run ID.
            scores (Dict[str, float]): Calculated scores for each run.

        Returns:
            Dict[str, List[Experience]]: Updated experiences with advantages broadcasted.
        """
        for run_id, exps in run_exps.items():
            score = scores[run_id]
            traj_length = len(exps)
            for exp in exps:
                exp.advantages = exp.action_mask * score  # type: ignore [operator]
                if self.enable_step_norm:
                    exp.advantages /= traj_length
                exp.returns = exp.advantages.clone()
        return run_exps

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if len(exps) == 0:
            return [], {}
        cnt = 0
        metric_list = []
        # Step 1: split the experiences into sub-groups by task
        task_exps = group_by(exps, "task")
        # Step 2: further split each task's experiences into sub-groups by run
        result_exps = []
        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")

            # Step3: extract the last experience (last step) from each run and calculate scores
            last_step_exps = {run_id: step_exps[-1] for run_id, step_exps in run_exps.items()}
            scores, metrics = self.calculate_last_step_advantage(last_step_exps)
            metric_list.append(metrics)

            # Step 4: broadcast the advantages to all previous steps
            run_exps = self.broadcast_advantages(run_exps, scores)
            for exps in run_exps.values():
                cnt += len(exps)
                result_exps.extend(exps)

        try:
            metrics = gather_metrics(metric_list, "group_advantages")
            metrics["experience_count"] = cnt
        except ValueError:
            metrics = {}  # empty metric list causes ValueError, ignore it
        return result_exps, metrics

    def __call__(self, exps, **kwargs):
        return self.process(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        """Whether the advantage should be computed in the trainer loop."""
        return False

    @classmethod
    def default_args(cls) -> Dict:
        """Return the default configuration for this strategy."""
        return {
            "epsilon": 1e-6,
            "enable_step_norm": False,
        }
