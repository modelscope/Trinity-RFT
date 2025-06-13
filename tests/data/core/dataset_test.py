# -*- coding: utf-8 -*-
"""Test cases for Storage modules."""
import os
import unittest

from trinity.common.config import DataPipelineConfig, FormatConfig, StorageConfig
from trinity.data.core.dataset import RewardSchema, RftDataset
from trinity.data.core.formatter import BoxedMathAnswerFormatter, RLHFFormatter


class TestRftDataset(unittest.TestCase):
    """Test cases for RftDataset"""

    def setUp(self) -> None:
        self.data_pipeline_config = DataPipelineConfig(
            input_buffers=[
                StorageConfig(
                    path=os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "..",
                        "..",
                        "test_data",
                        "test_10",
                    ),
                    raw=True,
                )
            ],
            format=FormatConfig(
                prompt_key="problem",
                response_key="solution",
                solution_key="solution",
            ),
        )
        self.data_pipeline_config_sample_level_setting = DataPipelineConfig(
            input_buffers=[
                StorageConfig(
                    path=os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "..",
                        "..",
                        "test_data",
                        "test_10_with_rewfn_workflow",
                    ),
                    raw=True,
                )
            ],
            format=FormatConfig(
                prompt_key="problem",
                response_key="solution",
                solution_key="solution",
                workflow_key="workflow",
                reward_fn_key="reward_fn",
            ),
        )

    def test_rft_dataset_init(self):
        dataset = RftDataset(
            data_pipeline_config=self.data_pipeline_config, reward_schema="default"
        )

        self.assertEqual(len(dataset), 10)
        self.assertIsInstance(dataset.reward_schema, RewardSchema)

    def test_format_dataset(self):
        dataset = RftDataset(
            data_pipeline_config=self.data_pipeline_config, reward_schema="default"
        )
        original_data = dataset.data
        # no formatter
        dataset.format(formatters=[])
        self.assertEqual(dataset.data, original_data)

        # apply formatters
        dataset.format(
            formatters=[
                BoxedMathAnswerFormatter(config=self.data_pipeline_config.format),
                RLHFFormatter(config=self.data_pipeline_config.format),
            ]
        )
        self.assertNotEqual(dataset.data, original_data)


if __name__ == "__main__":
    unittest.main()
