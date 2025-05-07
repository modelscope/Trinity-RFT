"""Tests for trainer."""
import os
import shutil
from abc import abstractmethod
from datetime import datetime

import ray

from tests.tools import (
    RayUnittestBase,
    TensorBoardParser,
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.cli.launcher import both
from trinity.common.constants import MonitorType, SyncMethod


class BaseTrainerCase(RayUnittestBase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.model.model_path = get_model_path()
        self.config.trainer.engine_type = "vllm_async"
        self.config.trainer.repeat_times = 3
        self.config.monitor.monitor_type = MonitorType.TENSORBOARD
        self.config.model.checkpoint_path = os.path.join(
            get_checkpoint_path(), f"train-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        self.config.synchronizer.sync_interval = 2
        self.config.synchronizer.sync_method = SyncMethod.NCCL
        self.config.explorer.eval_interval = 4
        self.config.trainer.eval_interval = 4

    @abstractmethod
    def test_trainer(self):
        """Test the trainer."""


class TestTrainerCountdown(BaseTrainerCase):
    def test_trainer(self):
        """Test the trainer."""
        self.config.data = get_unittest_dataset_config("countdown")
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.save_freq = 8
        both(self.config)
        # check tensorboard
        parser = TensorBoardParser(os.path.join(self.config.monitor.job_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 8)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 8)
        # check checkpoint
        from trinity.common.models.utils import get_checkpoint_dir_with_step_num

        checkpoint_dir = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.model.checkpoint_path,
            trainer_type=self.config.trainer.trainer_type,
            step_num=None,
        )
        self.assertTrue(os.path.exists(checkpoint_dir))
        self.assertTrue(checkpoint_dir.endswith("step_8"))

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.model.checkpoint_path)
