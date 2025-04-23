# -*- coding: utf-8 -*-
"""Test cases for Config modules."""
import os
import unittest

from trinity.common.config import load_config

config_yaml_path = os.path.join(os.path.dirname(__file__), "tmp", "template_config.yaml")


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config = load_config(config_yaml_path)
        self.assertIsNotNone(config.trainer.trainer_config)
        self.assertEqual(config.trainer.trainer_config.trainer.n_gpus_per_node, 4)
        self.assertEqual(config.trainer.trainer_config.trainer.nnodes, 1)
        self.assertEqual(config.trainer.trainer_config.trainer.project_name, config.monitor.project)
        self.assertEqual(config.trainer.trainer_config.trainer.experiment_name, config.monitor.name)
        self.assertEqual(config.trainer.trainer_config.trainer.project_name, config.monitor.project)
        self.assertEqual(
            config.trainer.trainer_config.trainer.save_freq,
            config.synchronizer.sync_iteration_interval,
        )

    def test_all_examples_are_valid(self): # TODO: useless
        # ../../scripts/config
        example_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "config")
        for filename in ["countdown", "gsm8k"]:
            if filename.endswith(".yaml"):
                config_path = os.path.join(example_dir, filename)
                try:
                    load_config(config_path)
                except Exception as e:
                    print(f"Error loading config {config_path}: {e}")
                    raise e
