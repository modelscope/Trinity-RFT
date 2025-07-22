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
from trinity.buffer import get_buffer_reader
from trinity.cli.launcher import explore


class BaseExplorerCase(RayUnittestBase):
    def setUp(self):
        self.config = get_template_config()
        self.config.buffer.total_epochs = 2
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.algorithm.repeat_times = 2
        self.config.monitor.monitor_type = "tensorboard"
        self.config.project = "Trinity-unittest"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 2
        self.config.explorer.eval_interval = 4

    @abstractmethod
    def test_explorer(self):
        """Test explorer"""


class TestExplorerCountdownEval(BaseExplorerCase):
    def test_explorer(self):
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.eval_tasksets.extend(
            [
                get_unittest_dataset_config("countdown", "test"),
                get_unittest_dataset_config("eval_short"),
                get_unittest_dataset_config("eval_long"),
            ]
        )
        self.config.name = f"explore-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.explorer.rollout_model.use_v1 = True
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)
        self.assertTrue("eval/eval_short/accuracy/max" in eval_metrics)
        self.assertTrue("eval/eval_long/accuracy/max" in eval_metrics)


class TestExplorerCountdownNoEval(BaseExplorerCase):
    def test_explorer(self):
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.name = f"explore-no-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.explorer.rollout_model.use_v1 = False
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) == 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)


class TestExplorerWithAddStrategy(BaseExplorerCase):
    def test_explorer(self):
        import ray

        from trinity.explorer.explorer import Explorer

        self.config.algorithm.repeat_times = 2
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.add_strategy = "random"
        self.config.name = f"explore-add-strategy-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # some step may be skipped due to same reward
        self.config.algorithm.add_strategy = "reward_variance"
        self.config.check_and_update()
        explorer = (
            ray.remote(Explorer)
            .options(
                name=self.config.explorer.name,
                namespace=ray.get_runtime_context().namespace,
            )
            .remote(self.config)
        )
        ray.get(explorer.prepare.remote())
        ray.get(explorer.sync_weight.remote())
        ray.get(explorer.explore.remote())
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) == 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        self.assertTrue(parser.metric_exist("rollout/experience_count"))
        experience_counts = parser.metric_values("rollout/experience_count")
        self.assertTrue(len(experience_counts) == 4)
        for count in experience_counts:
            self.assertTrue(count >= 0)
            self.assertTrue(count <= 2 * 4)  # repeat_times * batch_size
            self.assertTrue(count % 2 == 0)  # should be multiple of repeat_times

        reader = get_buffer_reader(
            self.config.buffer.trainer_input.experience_buffer, self.config.buffer
        )
        exps = []
        try:
            batch = reader.read()
            exps.extend(batch)
        except StopIteration:
            pass
        self.assertTrue(len(exps) <= 4 * 2 * 4)  # step * repeat_times * batch_size
        self.assertTrue(len(exps) % (2 * 4) == 0)  # should be multiple of repeat_times
        ray.get(explorer.shutdown.remote())
