"""Tests for explorer."""
import os
from abc import abstractmethod
from datetime import datetime

from tests.tools import (
    RayUnittestBase,
    TensorBoardParser,
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.cli.launcher import bench, explore
from trinity.common.constants import MonitorType


class BaseExplorerCase(RayUnittestBase):
    def setUp(self):
        self.config = get_template_config()
        self.config.global_config.total_epochs = 2
        self.config.global_config.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.engine_type = "vllm_async"
        self.config.explorer.repeat_times = 2
        self.config.monitor.monitor_type = MonitorType.TENSORBOARD
        self.config.monitor.project = "Trinity-unittest"
        self.config.model.checkpoint_path = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 2
        self.config.global_config.eval_interval = 4

    @abstractmethod
    def test_explorer(self):
        """Test explorer"""


class TestExplorerCountdownEval(BaseExplorerCase):
    def test_explorer(self):
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("countdown", "test")
        )
        self.config.monitor.name = f"explore-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.explorer.use_v1 = True
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.job_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)


class TestExplorerCountdownNoEval(BaseExplorerCase):
    def test_explorer(self):
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.monitor.name = f"explore-no-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.explorer.use_v1 = False
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.job_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) == 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)


class TestExplorerCountdownBench(BaseExplorerCase):
    def test_explorer(self):
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("countdown", "test")
        )
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("countdown", "test")
        )
        self.config.monitor.name = f"explore-bench-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.explorer.use_v1 = True
        self.config.check_and_update()
        bench(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.job_dir, "tensorboard"))
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
