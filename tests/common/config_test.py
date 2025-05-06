# -*- coding: utf-8 -*-
"""Test cases for Config modules."""
import os
import unittest

from tests.tools import get_template_config
from trinity.common.config import load_config


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config = get_template_config()
        config.check_and_update()
        self.assertIsNotNone(config.trainer.trainer_config)
        self.assertEqual(config.trainer.trainer_config.trainer.n_gpus_per_node, 2)
        self.assertEqual(config.trainer.trainer_config.trainer.nnodes, 1)
        self.assertEqual(config.trainer.trainer_config.trainer.project_name, config.monitor.project)
        self.assertEqual(config.trainer.trainer_config.trainer.experiment_name, config.monitor.name)
        self.assertEqual(config.trainer.trainer_config.trainer.project_name, config.monitor.project)
        self.assertEqual(
            config.trainer.trainer_config.trainer.save_freq,
            config.synchronizer.sync_interval,
        )

    def test_all_examples_are_valid(self):
        example_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        for example_name in os.listdir(example_dir):
            for filename in os.listdir(os.path.join(example_dir, example_name)):
                if filename.endswith(".yaml") and not filename.startswith("train"):
                    print(f"Checking config: {filename}")
                    config_path = os.path.join(example_dir, example_name, filename)
                    try:
                        load_config(config_path)
                    except Exception as e:
                        print(f"Error loading config {config_path}: {e}")
                        raise e
