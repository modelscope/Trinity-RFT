# -*- coding: utf-8 -*-
"""Test cases for data task parser."""
import unittest

import agentscope
from agentscope.models import DashScopeChatWrapper
from loguru import logger

from trinity.common.config import DataPipelineConfig
from trinity.data.controllers.task_parser import DataTaskParser


class TestTaskParser(unittest.TestCase):
    """Test cases for data task parser."""

    def setUp(self) -> None:
        print("setup", flush=True)

        api_key = "your_dashscope_key"

        agentscope.init(
            model_configs=[
                {
                    "config_name": "my-qwen-instruction",
                    "model_type": "dashscope_chat",
                    "model_name": "qwen2.5-72b-instruct",
                },
            ],
        )
        self.agent = DashScopeChatWrapper(
            config_name="_",
            model_name="qwen-max",
            api_key=api_key,
            stream=False,
        )

    def _run_test(self, rft_config, return_none=False):
        task_parser = DataTaskParser(rft_config, self.agent)

        dj_config, _, _, _ = task_parser.parse_to_dj_config()
        if return_none:
            self.assertIsNone(dj_config)
            logger.info("None dj config.")
        else:
            self.assertIsNotNone(dj_config)
            op_weights = {}
            for op in dj_config.process:
                op_name = list(op.keys())[0]
                op_weights[op_name] = op[op_name]["op_weight"]
            logger.info(op_weights)

    def test_instruction1(self):
        rft_config = DataPipelineConfig()
        rft_config.dj_process_desc = "Please recommend a data filtering strategy for me."
        self._run_test(rft_config)

    def test_instruction2(self):
        rft_config = DataPipelineConfig()
        rft_config.dj_process_desc = "Do nothing."
        self._run_test(rft_config, return_none=True)

    def test_instruction3(self):
        rft_config = DataPipelineConfig()
        rft_config.dj_process_desc = "Remove samples with repeat contents."
        self._run_test(rft_config)


if __name__ == "__main__":
    unittest.main()
