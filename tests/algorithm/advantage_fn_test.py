import unittest

import numpy as np
import torch

from trinity.algorithm.advantage_fn import ADVANTAGE_FN
from trinity.common.experience import EID, Experience


class TestGroupedAdvantageFn(unittest.TestCase):
    """Test cases for group-based advantage functions."""

    def test_grpo_advantage(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(**advantage_fn_cls.default_args())
        task_num = 3
        repeat_times = 5
        exps = [
            Experience(
                eid=EID(batch=0, task=j, run=i),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]

        # test group_epxeriences
        grouped_exps = advantage_fn.group_experiences(exps)
        self.assertEqual(len(grouped_exps), task_num)

        # test calculate_group_advantage
        for group_id, group_exps in grouped_exps.items():
            modified_exps, group_metrics = advantage_fn.calculate_group_advantage(
                group_id, group_exps
            )
            self.assertEqual(len(modified_exps), repeat_times)
            self.assertIn("reward_mean", group_metrics)
            self.assertIn("reward_std", group_metrics)

        # test the full pipeline

        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]
        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 2.0)
        self.assertTrue(
            metrics["group_advantages/reward_std/mean"]
            == torch.std(torch.tensor([i for i in range(repeat_times)], dtype=torch.float32)).item()
        )

        repeat_times = 1
        exps = [
            Experience(
                eid=EID(batch=0, task=j, run=i),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]
        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 0.0)
        self.assertTrue(metrics["group_advantages/reward_std/mean"] == 1.0)

    def test_grpo_reward_std(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-6, std_threshold=0.0)
        task_num = 3
        repeat_times = 5
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=0.5,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]

        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), 0)
        self.assertIn("group_advantages/skipped_count/mean", metrics)
        self.assertEqual(metrics["group_advantages/skipped_count/mean"], 5)

    def test_grpo_correct_bias(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-7, rank_penalty=0.2)
        task_num = 2
        repeat_times = 4
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                logprobs=torch.tensor([0.1 * i for _ in range(5)]),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]
        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertAlmostEqual(
            metrics["group_advantages/reward_mean/mean"],
            torch.mean(torch.tensor([0.0, 0.95, 1.80, 2.55], dtype=torch.float32)).item(),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["group_advantages/reward_std/mean"],
            torch.std(torch.tensor([0.0, 0.95, 1.80, 2.55], dtype=torch.float32)).item(),
            places=6,
        )

    def test_duplicate_grpo(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-6, std_threshold=0.0, duplicate_experiences=True)
        task_num = 3
        repeat_times = 5
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=np.random.rand(),
            )
            for i in range(repeat_times)
            for j in range(task_num - 1)
        ]
        zero_adv_exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=0.5,
            )
            for i in range(repeat_times)
            for j in range(task_num - 1, task_num * 2)
        ]
        exps.extend(zero_adv_exps)

        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), 2 * task_num * repeat_times)

        exps, metrics = advantage_fn(zero_adv_exps)
        self.assertEqual(len(exps), 0)

    def test_step_wise_grpo_advantage(self):
        advantage_fn_cls = ADVANTAGE_FN.get("step_wise_grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-7)
        self.assertEqual(advantage_fn.epsilon, 1e-7)
        task_num = 2
        repeat_times = 3
        step_num = 4
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                    step=k,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for k in range(step_num)
            for i in range(repeat_times)
            for j in range(task_num)
        ]

        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times * step_num)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 1.0)
        self.assertTrue(
            metrics["group_advantages/reward_std/mean"]
            == torch.std(torch.tensor([i for i in range(repeat_times)], dtype=torch.float32)).item()
        )
