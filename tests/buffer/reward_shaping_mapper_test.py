import unittest
from typing import List

import torch

from trinity.buffer.pipelines.experience_pipeline import ExperienceOperator
from trinity.common.config import OperatorConfig, RewardShapingConfig
from trinity.common.constants import OpType
from trinity.common.experience import EID, Experience


def get_experiences(task_num: int, repeat_times: int = 1, step_num: int = 1) -> List[Experience]:
    """Generate a list of experiences for testing."""
    return [
        Experience(
            eid=EID(task=i, run=j, step=k),
            tokens=torch.zeros((5,)),
            prompt_length=4,
            reward=j,
            logprobs=torch.tensor([0.1]),
            info={
                "quality_score": i,
                "difficulty_score": k,
            },
        )
        for i in range(task_num)
        for j in range(repeat_times)
        for k in range(step_num)
    ]


class TestRewardShapingMapper(unittest.TestCase):
    def test_experience_pipeline(self):
        # test input cache
        op_configs = [
            OperatorConfig(
                name="reward_shaping_mapper",
                args={
                    "reward_shaping_configs": [
                        RewardShapingConfig(
                            stats_key="quality_score",
                            op_type=OpType.ADD,
                            weight=1.0,
                        ),
                        RewardShapingConfig(
                            stats_key="difficulty_score",
                            op_type=OpType.MUL,
                            weight=0.5,
                        ),
                    ]
                },
            )
        ]
        ops = ExperienceOperator.create_operators(op_configs)
        self.assertEqual(len(ops), 1)

        op = ops[0]
        task_num = 8
        repeat_times = 4
        step_num = 2
        experiences = get_experiences(
            task_num=task_num, repeat_times=repeat_times, step_num=step_num
        )
        res_exps, metrics = op.process(experiences)
        self.assertEqual(len(res_exps), task_num * repeat_times * step_num)
        self.assertEqual(len(metrics), 0)

        for prev_exp, res_exp in zip(experiences, res_exps):
            self.assertEqual(
                (prev_exp.reward + prev_exp.info["quality_score"])
                * 0.5
                * prev_exp.info["difficulty_score"],
                res_exp.reward,
            )
