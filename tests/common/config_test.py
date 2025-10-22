# -*- coding: utf-8 -*-
"""Test cases for Config modules."""
import datetime
import os
import shutil
import unittest

from tests.tools import get_template_config
from trinity.common.config import InferenceModelConfig, load_config

CHECKPOINT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "temp_checkpoint_dir")


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
            config.buffer.explorer_input.taskset.repeat_times, config.algorithm.repeat_times
        )
        self.assertEqual(config.model.model_path, config.model.critic_model_path)
        self.assertEqual(config.model.model_path, config.explorer.rollout_model.model_path)

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
        config = get_template_config()
        config.name = "test"
        config.project = "unittest"
        config.checkpoint_root_dir = CHECKPOINT_ROOT_DIR

        dir_path = os.path.join(config.checkpoint_root_dir, config.project, config.name)
        os.makedirs(os.path.join(dir_path, "global_step_1"))

        config.continue_from_checkpoint = True
        config.check_and_update()
        self.assertEqual(config.name, "test")

        config.continue_from_checkpoint = False
        config.check_and_update()
        self.assertTrue(config.name.startswith("test_"))
        timestamp = config.name.split("_")[-1]
        self.assertTrue(datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S"))

    def test_config_flatten(self):
        config = get_template_config()
        flat_config = config.flatten()
        self.assertIsInstance(flat_config, dict)
        for key, value in flat_config.items():
            self.assertIsInstance(key, str)
            self.assertNotIsInstance(value, dict)

    def test_update_config_from_ray_cluster(self):
        config = get_template_config()
        config.cluster.node_num = None
        config.cluster.gpu_per_node = None

        config._update_config_from_ray_cluster()
        self.assertEqual(config.cluster.node_num, 2)
        self.assertEqual(config.cluster.gpu_per_node, 2)

    def tearDown(self):
        if os.path.exists(CHECKPOINT_ROOT_DIR):
            shutil.rmtree(CHECKPOINT_ROOT_DIR)
