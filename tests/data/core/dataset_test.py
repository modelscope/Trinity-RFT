# -*- coding: utf-8 -*-
"""Test cases for Storage modules."""
import os
import unittest

from trinity.common.config import DataProcessorConfig, FormatConfig
from trinity.common.rewards import AccuracyReward
from trinity.common.task import TaskSet
from trinity.common.workflows import MathWorkflow, SimpleWorkflow
from trinity.data.core.dataset import RewardSchema, RftDataset
from trinity.data.core.formatter import BoxedMathAnswerFormatter, RLHFFormatter


class TestRftDataset(unittest.TestCase):
    """Test cases for RftDataset"""

    def setUp(self) -> None:
        self.data_config = DataProcessorConfig(
            source_data_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "test_data",
                "test_10",
            ),
            format=FormatConfig(
                prompt_key="problem",
                response_key="solution",
                solution_key="solution",
            ),
        )
        self.data_config_sample_level_setting = DataProcessorConfig(
            source_data_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "test_data",
                "test_10_with_rewfn_workflow",
            ),
            format=FormatConfig(
                prompt_key="problem",
                response_key="solution",
                solution_key="solution",
                workflow_key="workflow",
                reward_fn_key="reward_fn",
            ),
        )

    def test_rft_dataset_init(self):
        dataset = RftDataset(data_config=self.data_config, reward_schema="default")

        self.assertEqual(len(dataset), 10)
        self.assertIsInstance(dataset.reward_schema, RewardSchema)

    def test_format_dataset(self):
        dataset = RftDataset(data_config=self.data_config, reward_schema="default")
        original_data = dataset.data
        # no formatter
        dataset.format(formatters=[])
        self.assertEqual(dataset.data, original_data)

        # apply formatters
        dataset.format(
            formatters=[
                BoxedMathAnswerFormatter(config=self.data_config.format),
                RLHFFormatter(config=self.data_config.format),
            ]
        )
        self.assertNotEqual(dataset.data, original_data)

    def test_to_taskset(self):
        dataset = RftDataset(data_config=self.data_config, reward_schema="default")
        taskset = dataset.to_taskset()
        self.assertIsInstance(taskset, TaskSet)
        self.assertEqual(len(taskset), 10)
        self.assertIsNone(taskset.reward_fn)
        self.assertIsNone(taskset.workflow)
        self.assertEqual(taskset._index, 0)

    def test_to_taskset_with_global_settings(self):
        dataset = RftDataset(data_config=self.data_config, reward_schema="default")
        taskset = dataset.to_taskset(
            reward_fn=AccuracyReward,
            workflow=SimpleWorkflow,
        )
        self.assertIsInstance(taskset, TaskSet)
        self.assertEqual(taskset.workflow, SimpleWorkflow)
        self.assertEqual(taskset.reward_fn, AccuracyReward)

    def test_to_taskset_with_sample_level_settings(self):
        dataset = RftDataset(
            data_config=self.data_config_sample_level_setting, reward_schema="default"
        )
        taskset = dataset.to_taskset()
        self.assertIsInstance(taskset, TaskSet)
        for task in taskset.tasks:
            self.assertEqual(task.workflow, MathWorkflow)
            self.assertEqual(task.reward_fn, AccuracyReward)

    def test_to_taskset_with_both_settings(self):
        dataset = RftDataset(
            data_config=self.data_config_sample_level_setting, reward_schema="default"
        )
        taskset = dataset.to_taskset(
            reward_fn=AccuracyReward,
            workflow=SimpleWorkflow,
        )
        self.assertIsInstance(taskset, TaskSet)
        for task in taskset.tasks:
            self.assertEqual(task.workflow, MathWorkflow)
            self.assertEqual(task.reward_fn, AccuracyReward)
        self.assertEqual(taskset.workflow, SimpleWorkflow)
        self.assertEqual(taskset.reward_fn, AccuracyReward)


if __name__ == "__main__":
    unittest.main()
