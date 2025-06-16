# -*- coding: utf-8 -*-
"""Test cases for data task parser."""
import os
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

        api_key = os.environ.get("OPENAI_API_KEY", None)

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
