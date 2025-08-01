import asyncio
from typing import Dict, List, Tuple

import torch

from trinity.algorithm.add_strategy.add_strategy import (
    ADD_STRATEGY,
    AddStrategy,
    group_by,
)
from trinity.buffer import BufferWriter
from trinity.common.experience import Experience
from trinity.utils.monitor import gather_metrics


@ADD_STRATEGY.register_module("step_wise_grpo")
class StepWiseGRPOStrategy(AddStrategy):
    """
    An example AddStrategy that broadcasts advantages from the last step to previous steps.
    Inspired by rLLM (https://github.com/rllm-org/rllm).
    """

    def __init__(
        self,
        writer: BufferWriter,
        epsilon: float = 1e-6,
        enable_step_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(writer)
        self.epsilon = epsilon
        self.enable_step_norm = enable_step_norm

    def calculate_group_advantage(
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

    async def add(self, exps: List[Experience], step: int) -> Tuple[int, Dict]:
        if len(exps) == 0:
            return 0, {}
        cnt = 0
        tasks = []
        metric_list = []
        # Step 1: split the experiences into sub-groups by task
        task_exps = group_by(exps, "task")
        # Step 2: further split each task's experiences into sub-groups by run
        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")

            # Step3: extract the last experience (last step) from each run and calculate scores
            last_step_exps = {run_id: step_exps[-1] for run_id, step_exps in run_exps.items()}
            scores, metrics = self.calculate_group_advantage(last_step_exps)
            metric_list.append(metrics)

            # Step 4: broadcast the advantages to all previous steps
            run_exps = self.broadcast_advantages(run_exps, scores)
            for exps in run_exps.values():
                cnt += len(exps)
                tasks.append(self.writer.write_async(exps))

        if tasks:
            await asyncio.gather(*tasks)
        try:
            metrics = gather_metrics(metric_list, "group_advantages")
        except ValueError:
            metrics = {}  # empty metric list causes ValueError, ignore it
        return cnt, metrics

    @classmethod
    def default_args(cls) -> Dict:
        """Return the default configuration for this strategy."""
        return {
            "epsilon": 1e-6,
            "enable_step_norm": False,
        }
