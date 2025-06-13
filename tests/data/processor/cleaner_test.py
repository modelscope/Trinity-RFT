# -*- coding: utf-8 -*-
"""Test cases for cleaner."""
import unittest

from trinity.common.config import load_config
from trinity.data.controllers.task_parser import DataTaskParser
from trinity.data.core.dataset import RftDataset
from trinity.data.processors.cleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):
    """Test cases for data cleaner."""

    def setUp(self) -> None:
        print("setup", flush=True)

        self.rft_config = load_config("./tests/test_configs/cleaner_test_rft_cfg.yaml")
        # print(self.rft_config)
        self.ds_list = [
            {"text": "Today is"},
            {"text": "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"},
            {"text": "a v s e c s f e f g a a a  "},
            {"text": "，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►"},
            {"text": "中文也是一个字算一个长度"},
        ]

    def _run_test(self, tgt_list, weight=1, data_dist="gaussian"):
        task_parser = DataTaskParser(self.rft_config.data_processor.task_pipeline)
        dj_config, _, _, _ = task_parser.parse_to_dj_config()
        op_weights = {}
        for op_config in dj_config.process:
            op_name, _ = list(op_config.items())[0]
            op_weights[op_name] = weight

        cleaner = DataCleaner(
            dj_config,
            clean_strategy="iterative",
            min_size_ratio=self.rft_config.data_processor.task_pipeline.min_size_ratio,
            data_dist=data_dist,
            op_weights=op_weights,
        )

        dataset = RftDataset(self.rft_config.data_processor.task_pipeline)
        dataset.read_from_buffer()
        dataset = cleaner.process([dataset])

        res_list = dataset.data.select_columns("text").to_list()
        print(res_list)
        self.assertEqual(res_list, tgt_list)
        self.assertNotIn("clean_email_mapper", cleaner.dj_cfg.process)

    def test_dj_executor(self):
        tgt_list = [
            {"text": "Today is"},
            {"text": "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"},
            {"text": "a v s e c s f e f g a a a  "},
            {"text": "中文也是一个字算一个长度"},
        ]

        self.rft_config.data_processor.task_pipeline.min_size_ratio = None

        self._run_test(tgt_list)

    def test_iterative_clean(self):
        tgt_list = [
            {"text": "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"},
            {"text": "a v s e c s f e f g a a a  "},
        ]

        self.rft_config.data_processor.task_pipeline.min_size_ratio = 0.5

        self._run_test(tgt_list)

    def test_weight(self):
        tgt_list = [
            {"text": "Today is"},
            {"text": "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"},
            {"text": "a v s e c s f e f g a a a  "},
        ]

        self.rft_config.data_processor.task_pipeline.min_size_ratio = 0.5

        self._run_test(tgt_list, weight=0.5)

    def test_uniform_dist(self):
        tgt_list = []

        self.rft_config.data_processor.task_pipeline.min_size_ratio = 0.5

        self._run_test(tgt_list, data_dist="uniform")


if __name__ == "__main__":
    unittest.main()
