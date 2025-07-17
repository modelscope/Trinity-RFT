# -*- coding: utf-8 -*-
"""Test cases for Config modules."""
import datetime
import os
import shutil
import unittest

from tests.tools import get_template_config
from trinity.common.config import InferenceModelConfig, load_config


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config = get_template_config()
        config.buffer.batch_size = 8
        config.algorithm.repeat_times = 10
        config.model.model_path = "Qwen/Qwen3-1.7B"
        config.cluster.gpu_per_node = 8
        config.cluster.node_num = 2
        config.explorer.rollout_model.engine_num = 2
        config.explorer.rollout_model.tensor_parallel_size = 2
        config.explorer.auxiliary_models.append(
            InferenceModelConfig(model_path="Qwen/Qwen3-32B", tensor_parallel_size=4, engine_num=1),
        )
        config.check_and_update()
        self.assertIsNotNone(config.trainer.trainer_config)
        self.assertEqual(config.trainer.trainer_config.trainer.n_gpus_per_node, 8)
        self.assertEqual(config.trainer.trainer_config.trainer.nnodes, 1)
        self.assertEqual(config.trainer.trainer_config.trainer.project_name, config.project)
        self.assertEqual(config.trainer.trainer_config.trainer.experiment_name, config.name)
        self.assertEqual(
            config.buffer.explorer_input.taskset.rollout_args.n, config.algorithm.repeat_times
        )
        self.assertEqual(config.model.model_path, config.model.critic_model_path)
        self.assertEqual(config.model.model_path, config.explorer.rollout_model.model_path)
        self.assertEqual(
            config.trainer.trainer_config.trainer.save_freq,
            config.synchronizer.sync_interval,
        )

    def test_all_examples_are_valid(self):
        example_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
        for example_name in os.listdir(example_dir):
            for filename in os.listdir(os.path.join(example_dir, example_name)):
                if filename.endswith(".yaml") and not (
                    filename.startswith("train_")
                    or filename.startswith("verl_")
                    or filename.startswith("dj_")
                ):
                    print(f"Checking config: {filename}")
                    config_path = os.path.join(example_dir, example_name, filename)
                    try:
                        config = load_config(config_path)
                        config.checkpoint_root_dir = "./.cache/"
                        config.check_and_update()
                    except Exception as e:
                        print(f"Error loading config {config_path}: {e}")
                        raise e

    def test_continue_from_checkpoint_is_valid(self):
        self.config = get_template_config()
        self.config.name = "test"
        self.config.project = "unittest"
        self.config.checkpoint_root_dir = os.path.join(
            os.path.dirname(__file__), "temp_checkpoint_dir"
        )

        dir_path = os.path.join(
            self.config.checkpoint_root_dir, self.config.project, self.config.name
        )
        os.makedirs(os.path.join(dir_path, "global_step_1"))

        self.config.continue_from_checkpoint = True
        self.config.check_and_update()
        self.assertEqual(self.config.name, "test")

        self.config.continue_from_checkpoint = False
        self.config.check_and_update()
        self.assertTrue(self.config.name.startswith("test_"))
        timestamp = self.config.name.split("_")[-1]
        self.assertTrue(datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S"))

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_root_dir)
