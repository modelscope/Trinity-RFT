# -*- coding: utf-8 -*-
"""Test cases for Storage modules."""
import os
import unittest

from trinity.common.config import DataProcessorConfig, FormatConfig
from trinity.data.core.dataset import RftDataset
from trinity.data.core.formatter import (
    BoxedMathAnswerFormatter,
    ComposedFormatter,
    RewardFormatter,
    RLHFFormatter,
    SFTFormatter,
)


class TestBoxedMathDataset(unittest.TestCase):
    """Test cases for RftDataset"""

    def setUp(self) -> None:
        self.data_config = DataProcessorConfig(
            raw_data_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "test_data",
                "test_10",
            ),
            format_config=FormatConfig(
                prompt_key="problem",
                response_key="answer",
                solution_key="solution",
                chat_template="User: {}\nAssistant: ",
            ),
        )

    def test_init(self):
        formatter = BoxedMathAnswerFormatter(config=self.data_config.format_config)
        # test for existing configs
        self.assertEqual(formatter.config.prompt_key, "problem")
        self.assertEqual(formatter.config.response_key, "answer")
        self.assertEqual(formatter.config.solution_key, "solution")
        self.assertEqual(formatter.config.chat_template, "User: {}\nAssistant: ")
        # test for default configs
        self.assertEqual(formatter.config.reward_key, "")
        self.assertEqual(formatter.config.chosen_key, "")
        self.assertEqual(formatter.config.rejected_key, "")
        self.assertEqual(formatter.config.label_key, "")

    def test_transform(self):
        dataset = RftDataset(data_config=self.data_config, reward_schema="default")
        formatter = BoxedMathAnswerFormatter(config=self.data_config.format_config)
        self.assertNotIn(formatter.config.response_key, dataset.data.column_names)
        dataset.format(formatter)
        self.assertIn(formatter.config.response_key, dataset.data.column_names)


class TestRLHFFormatter(unittest.TestCase):
    """Test cases for RLHFFormatter"""

    def setUp(self) -> None:
        self.data_config = DataProcessorConfig(
            raw_data_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "test_data",
                "test_10",
            ),
            format_config=FormatConfig(
                prompt_key="problem",
                chat_template="User: {}\nAssistant: ",
            ),
        )

    def test_render_template(self):
        sample = {
            "problem": "What is the capital of France?",
        }
        formatter = RLHFFormatter(config=self.data_config.format_config)
        res_sample = formatter._render_template(sample)
        self.assertEqual(
            res_sample[formatter.config.prompt_key],
            "User: What is the capital of France?\nAssistant: ",
        )

    def test_render_template_without_chat_template(self):
        self.data_config.format_config.chat_template = ""
        sample = {
            "problem": "What is the capital of France?",
        }
        formatter = RLHFFormatter(config=self.data_config.format_config)
        res_sample = formatter._render_template(sample)
        self.assertEqual(res_sample[formatter.config.prompt_key], "What is the capital of France?")
        self.data_config.format_config.chat_template = "User: {}\nAssistant: "

    def test_render_template_with_tokenizer(self):
        # TODO
        pass

    def test_render_template_with_tokenizer_and_chat_format(self):
        # TODO
        pass


class TestRewardFormatter(unittest.TestCase):
    """Test cases for RewardFormatter"""

    def setUp(self) -> None:
        self.data_config = DataProcessorConfig(
            raw_data_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "test_data",
                "test_10",
            ),
            format_config=FormatConfig(
                prompt_key="problem",
                chosen_key="chosen",
                rejected_key="rejected",
                chat_template="User: {}\nAssistant: ",
            ),
        )

    def test_render_template(self):
        sample = {
            "problem": "What is the capital of France?",
            "chosen": "Paris",
            "rejected": "London",
        }
        formatter = RewardFormatter(config=self.data_config.format_config)
        res_sample = formatter._render_template(sample)
        self.assertEqual(
            res_sample[formatter.config.prompt_key],
            "User: What is the capital of France?\nAssistant: ",
        )
        self.assertEqual(res_sample[formatter.config.chosen_key], "Paris")
        self.assertEqual(res_sample[formatter.config.rejected_key], "London")

    def test_render_template_without_chat_template(self):
        self.data_config.format_config.chat_template = ""
        sample = {
            "problem": "What is the capital of France?",
            "chosen": "Paris",
            "rejected": "London",
        }
        formatter = RewardFormatter(config=self.data_config.format_config)
        res_sample = formatter._render_template(sample)
        self.assertEqual(res_sample[formatter.config.prompt_key], "What is the capital of France?")
        self.assertEqual(res_sample[formatter.config.chosen_key], "Paris")
        self.assertEqual(res_sample[formatter.config.rejected_key], "London")

    def test_render_template_with_tokenizer(self):
        # TODO
        pass

    def test_render_template_with_tokenizer_and_chat_format(self):
        # TODO
        pass


class TestSFTFormatter(unittest.TestCase):
    """Test cases for SFTFormatter"""

    def setUp(self) -> None:
        self.data_config = DataProcessorConfig(
            raw_data_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "test_data",
                "test_10",
            ),
            format_config=FormatConfig(
                prompt_key="problem",
                response_key="answer",
                chat_template="User: {}\nAssistant: ",
            ),
        )

    def test_render_template(self):
        sample = {
            "problem": "What is the capital of France?",
            "answer": "Paris",
        }
        formatter = SFTFormatter(config=self.data_config.format_config)
        res_sample = formatter._render_template(sample)
        self.assertEqual(
            res_sample[formatter.config.prompt_key],
            "User: What is the capital of France?\nAssistant: ",
        )
        self.assertEqual(res_sample[formatter.config.response_key], "Paris")

    def test_render_template_without_chat_template(self):
        self.data_config.format_config.chat_template = ""
        sample = {
            "problem": "What is the capital of France?",
            "answer": "Paris",
        }
        formatter = SFTFormatter(config=self.data_config.format_config)
        res_sample = formatter._render_template(sample)
        self.assertEqual(res_sample[formatter.config.prompt_key], "What is the capital of France?")
        self.assertEqual(res_sample[formatter.config.response_key], "Paris")
        self.data_config.format_config.chat_template = "User: {}\nAssistant: "

    def test_render_template_with_tokenizer(self):
        # TODO
        pass

    def test_render_template_with_tokenizer_and_chat_format(self):
        # TODO
        pass


class TestComposedFormatter(unittest.TestCase):
    """Test cases for ComposedFormatter"""

    def setUp(self) -> None:
        self.data_config = DataProcessorConfig(
            raw_data_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "..",
                "test_data",
                "test_10",
            ),
            format_config=FormatConfig(
                prompt_key="problem",
                response_key="answer",
                solution_key="solution",
                chat_template="User: {}\nAssistant: ",
            ),
        )
        self.sample = {
            "problem": "What is the capital of France?",
            "solution": "It's \\boxed{Paris}",
        }

    def test_compose(self):
        composed_formatter = ComposedFormatter(
            formatters=[
                BoxedMathAnswerFormatter(config=self.data_config.format_config),
                SFTFormatter(config=self.data_config.format_config),
            ]
        )
        self.assertNotIn(composed_formatter.config.response_key, self.sample)
        res_sample = composed_formatter.transform(self.sample)
        self.assertEqual(
            res_sample[composed_formatter.config.prompt_key],
            "User: What is the capital of France?\nAssistant: ",
        )
        self.assertIn(composed_formatter.config.response_key, res_sample)
        self.assertEqual(res_sample[composed_formatter.config.response_key], "Paris")
        self.assertEqual(res_sample[composed_formatter.config.solution_key], "It's \\boxed{Paris}")


if __name__ == "__main__":
    unittest.main()
