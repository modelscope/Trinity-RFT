# -*- coding: utf-8 -*-
"""Test cases for Synchronizer modules."""

import asyncio
import multiprocessing
import os
import shutil
import time
import unittest
from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import ray
from parameterized import parameterized_class

from tests.tools import (
    TensorBoardParser,
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.algorithm.algorithm import ALGORITHM_TYPE
from trinity.cli.launcher import both, explore, train
from trinity.common.config import Config, ExperienceBufferConfig
from trinity.common.constants import StorageType, SyncMethod, SyncStyle
from trinity.explorer.explorer import Explorer
from trinity.trainer.trainer import Trainer
from trinity.utils.log import get_logger

logger = get_logger(__name__)
CHECKPOINT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "temp_checkpoint_dir")


def trainer_monkey_patch(config: Config, max_steps: int, intervals: List[int]):
    async def new_sample_data(self):
        self.logger.info(f"Sample data for step {self.train_step_num + 1} started.")
        await asyncio.sleep(0.1)
        time.sleep(intervals[self.engine.global_steps - 1])
        self.logger.info(f"Sample data for step {self.train_step_num + 1} finished.")
        return [], {}, []

    async def new_train_step(self, exps) -> Dict:
        self.engine.algorithm = ALGORITHM_TYPE.get(config.algorithm.algorithm_type)
        self.engine.global_steps += 1
        self.logger.info(f"Training at step {self.engine.global_steps} started.")
        await asyncio.sleep(0.1)
        time.sleep(intervals[self.engine.global_steps - 1])
        metrics = {"actor/step": self.engine.global_steps}
        self.logger.info(f"Training at step {self.engine.global_steps} finished.")
        return metrics

    Trainer.train_step = new_train_step
    Trainer._sample_data = new_sample_data


def explorer_monkey_patch(config: Config, max_steps: int, intervals: List[int]):
    async def new_explore_step(self):
        if self.explore_step_num == max_steps:
            await self.save_checkpoint(sync_weight=False)
        self.explore_step_num += 1
        return self.explore_step_num <= max_steps

    def wrapper(old_save_checkpoint):
        async def new_save_checkpoint(self, sync_weight: bool = False):
            await asyncio.sleep(intervals.pop(0))
            await old_save_checkpoint(self, sync_weight)

        return new_save_checkpoint

    async def new_finish_explore_step(self, step: int, model_version: int) -> None:
        metric = {"rollout/model_version": model_version}
        self.monitor.log(metric, step=step)

    Explorer.explore_step = new_explore_step
    Explorer.save_checkpoint = wrapper(Explorer.save_checkpoint)
    Explorer._finish_explore_step = new_finish_explore_step


def run_trainer(config: Config, max_steps: int, intervals: List[int]) -> None:
    ray.init(ignore_reinit_error=True, namespace=config.ray_namespace)
    try:
        trainer_monkey_patch(config, max_steps, intervals)
        train(config)
    finally:
        ray.shutdown(_exiting_interpreter=True)


def run_explorer(config: Config, max_steps: int, intervals: List[int]) -> None:
    ray.init(ignore_reinit_error=True, namespace=config.ray_namespace)
    try:
        explorer_monkey_patch(config, max_steps, intervals)
        explore(config)
    finally:
        ray.shutdown(_exiting_interpreter=True)


def run_both(
    config: Config, max_steps: int, trainer_intervals: List[int], explorer_intervals: List[int]
) -> None:
    ray.init(ignore_reinit_error=True, namespace=config.ray_namespace)
    try:
        trainer_monkey_patch(config, max_steps, trainer_intervals)
        explorer_monkey_patch(config, max_steps, explorer_intervals)
        both(config)
    finally:
        ray.shutdown(_exiting_interpreter=True)


class BaseTestSynchronizer(unittest.TestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

    def tearDown(self):
        checkpoint_path = get_checkpoint_path()
        shutil.rmtree(os.path.join(checkpoint_path, "unittest"))


class TestSynchronizerExit(BaseTestSynchronizer):
    def test_synchronizer(self):
        config = get_template_config()
        config.project = "unittest"
        config.name = f"test_synchronizer_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        config.checkpoint_root_dir = get_checkpoint_path()
        config.buffer.total_epochs = 1
        config.buffer.batch_size = 4
        config.cluster.gpu_per_node = 2
        config.cluster.node_num = 1
        config.model.model_path = get_model_path()
        config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE,
        )
        config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        config.synchronizer.sync_style = SyncStyle.DYNAMIC_BY_EXPLORER
        config.synchronizer.sync_interval = 2
        config.trainer.save_interval = 100
        config.monitor.monitor_type = "tensorboard"
        trainer_config = deepcopy(config)
        trainer_config.mode = "train"
        trainer_config.buffer.train_batch_size = 4
        trainer_config.check_and_update()

        explorer1_config = deepcopy(config)
        explorer1_config.mode = "explore"
        explorer1_config.explorer.name = "explorer1"
        explorer1_config.explorer.rollout_model.engine_num = 1
        explorer1_config.explorer.rollout_model.tensor_parallel_size = 1
        explorer1_config.buffer.explorer_output = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE,
        )
        explorer1_config.check_and_update()

        trainer_process = multiprocessing.Process(
            target=run_trainer, args=(trainer_config, 8, [2, 1, 2, 1, 2, 1, 2, 1])
        )
        trainer_process.start()
        ray.init(ignore_reinit_error=True)
        while True:
            try:
                synchronizer = ray.get_actor("synchronizer", namespace=trainer_config.ray_namespace)
                break
            except ValueError:
                print("waiting for trainer to start.")
                time.sleep(5)
        explorer_process_1 = multiprocessing.Process(
            target=run_explorer,
            args=(explorer1_config, 8, [0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]),
        )
        explorer_process_1.start()

        self.assertEqual(
            synchronizer, ray.get_actor("synchronizer", namespace=trainer_config.ray_namespace)
        )
        for _ in range(12):  # Wait for up to 60 seconds
            try:
                explorer1 = ray.get_actor("explorer1", namespace=trainer_config.ray_namespace)
                ray.get(explorer1.is_alive.remote())
                break
            except ValueError:
                print("waiting for explorer1 to start.")
                time.sleep(5)
        trainer_process.terminate()
        trainer_process.join()
        self.assertEqual(
            synchronizer, ray.get_actor("synchronizer", namespace=trainer_config.ray_namespace)
        )

        explorer_process_1.terminate()
        explorer_process_1.join()
        time.sleep(6)
        with self.assertRaises(ValueError):
            ray.get_actor("synchronizer", namespace=trainer_config.ray_namespace)


@parameterized_class(
    (
        "sync_method",
        "sync_style",
        "max_steps",
        "trainer_intervals",
        "explorer1_intervals",
        "explorer2_intervals",
    ),
    [
        (
            SyncMethod.CHECKPOINT,
            SyncStyle.FIXED,
            8,
            [2, 1, 2, 1, 2, 1, 2, 1],
            [0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
            [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
        (
            SyncMethod.CHECKPOINT,
            SyncStyle.DYNAMIC_BY_EXPLORER,
            8,
            [2, 1, 2, 1, 2, 1, 2, 1],
            [0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
            [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
        (
            SyncMethod.MEMORY,
            SyncStyle.FIXED,
            8,
            [2, 1, 2, 1, 2, 1, 2, 1],
            [0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
            [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
        (
            SyncMethod.MEMORY,
            SyncStyle.DYNAMIC_BY_EXPLORER,
            8,
            [2, 1, 2, 1, 2, 1, 2, 1],
            [0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
            [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
    ],
)
class TestStateDictBasedSynchronizer(BaseTestSynchronizer):
    def test_synchronizer(self):
        config = get_template_config()
        config.project = "unittest"
        config.name = f"test_synchronizer_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        config.checkpoint_root_dir = get_checkpoint_path()
        config.buffer.total_epochs = 1
        config.buffer.batch_size = 4
        config.cluster.gpu_per_node = 2
        config.cluster.node_num = 1
        config.model.model_path = get_model_path()
        config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE,
        )
        config.synchronizer.sync_method = self.sync_method
        config.synchronizer.sync_style = self.sync_style
        config.synchronizer.sync_interval = 2
        config.trainer.save_interval = 100
        config.trainer.total_steps = self.max_steps
        config.monitor.monitor_type = "tensorboard"
        trainer_config = deepcopy(config)
        trainer_config.mode = "train"
        trainer_config.buffer.train_batch_size = 4
        trainer_config.check_and_update()

        explorer1_config = deepcopy(config)
        explorer1_config.mode = "explore"
        explorer1_config.explorer.name = "explorer1"
        explorer1_config.explorer.rollout_model.engine_num = 1
        explorer1_config.explorer.rollout_model.tensor_parallel_size = 1
        explorer1_config.buffer.explorer_output = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE,
        )
        explorer2_config = deepcopy(explorer1_config)
        explorer2_config.explorer.name = "explorer2"
        explorer1_config.check_and_update()
        explorer2_config.check_and_update()

        trainer_process = multiprocessing.Process(
            target=run_trainer, args=(trainer_config, self.max_steps, self.trainer_intervals)
        )
        trainer_process.start()
        ray.init(ignore_reinit_error=True)
        while True:
            try:
                ray.get_actor("queue-exp_buffer", namespace=trainer_config.ray_namespace)
                break
            except ValueError:
                print("waiting for trainer to start.")
                time.sleep(5)
        explorer_process_1 = multiprocessing.Process(
            target=run_explorer,
            args=(explorer1_config, self.max_steps, self.explorer1_intervals),
        )
        explorer_process_1.start()
        explorer_process_2 = multiprocessing.Process(
            target=run_explorer, args=(explorer2_config, self.max_steps, self.explorer2_intervals)
        )
        explorer_process_2.start()

        explorer_process_1.join(timeout=200)
        explorer_process_2.join(timeout=200)
        trainer_process.join(timeout=200)

        # check the tensorboard
        parser = TensorBoardParser(
            os.path.join(trainer_config.monitor.cache_dir, "tensorboard", "trainer")
        )
        actor_metrics = parser.metric_list("actor")
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 8)
        parser = TensorBoardParser(
            os.path.join(explorer1_config.monitor.cache_dir, "tensorboard", "explorer1")
        )
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        parser = TensorBoardParser(
            os.path.join(explorer2_config.monitor.cache_dir, "tensorboard", "explorer2")
        )
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)


@parameterized_class(
    ("sync_style", "max_steps", "trainer_intervals", "explorer_intervals"),
    [
        (
            SyncStyle.FIXED,
            8,
            [2, 1, 2, 1, 2, 1, 2, 1],
            [0, 2.5, 2.5, 2.5, 2.5, 0],
        ),
        (
            SyncStyle.DYNAMIC_BY_EXPLORER,
            8,
            [2, 1, 2, 1, 2, 1, 2, 1],
            [0, 0.5, 0.5, 0.5, 0.5, 0],
        ),
    ],
)
class TestNCCLBasedSynchronizer(BaseTestSynchronizer):
    def test_synchronizer(self):
        config = get_template_config()
        config.project = "unittest"
        config.name = f"test_synchronizer_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        config.checkpoint_root_dir = get_checkpoint_path()
        config.buffer.total_epochs = 1
        config.buffer.batch_size = 4
        config.trainer.total_steps = self.max_steps
        config.model.model_path = get_model_path()
        config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE,
        )
        config.synchronizer.sync_method = SyncMethod.NCCL
        config.synchronizer.sync_style = self.sync_style
        config.synchronizer.sync_interval = 2
        config.trainer.save_interval = 100
        config.monitor.monitor_type = "tensorboard"
        config.mode = "both"
        config.check_and_update()

        # TODO: test more interval cases
        both_process = multiprocessing.Process(
            target=run_both,
            args=(config, self.max_steps, self.trainer_intervals, self.explorer_intervals),
        )
        both_process.start()
        both_process.join(timeout=200)

        # check the tensorboard
        parser = TensorBoardParser(os.path.join(config.monitor.cache_dir, "tensorboard", "trainer"))
        actor_metrics = parser.metric_list("actor")
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 8)
        parser = TensorBoardParser(
            os.path.join(config.monitor.cache_dir, "tensorboard", "explorer")
        )
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)

    def tearDown(self):
        if os.path.exists(CHECKPOINT_ROOT_DIR):
            shutil.rmtree(CHECKPOINT_ROOT_DIR)
