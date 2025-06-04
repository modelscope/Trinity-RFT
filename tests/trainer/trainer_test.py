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
from trinity.cli.launcher import bench, both, train
from trinity.common.constants import AlgorithmType, MonitorType, SyncMethod


class BaseTrainerCase(RayUnittestBase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.buffer.total_epochs = 2
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.algorithm.repeat_times = 3
        self.config.explorer.rollout_model.use_v1 = False
        self.config.project = "Trainer-unittest"
        self.config.name = f"trainer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.monitor.monitor_type = MonitorType.TENSORBOARD
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 2
        self.config.synchronizer.sync_method = SyncMethod.NCCL
        self.config.explorer.eval_interval = 4

    @abstractmethod
    def test_trainer(self):
        """Test the trainer."""


class TestTrainerCountdown(BaseTrainerCase):
    def test_trainer(self):
        """Test the both and bench mode."""
        # test both mode
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("countdown", "test")
        )
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("copy_countdown", "test")
        )
        self.config.trainer.save_interval = 4
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        self.config.trainer.trainer_config.trainer.max_critic_ckpt_to_keep = 2
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
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
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint
        from trinity.common.models.utils import get_checkpoint_dir_with_step_num

        checkpoint_step_4 = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
            step_num=4,
        )
        checkpoint_step_8 = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
            step_num=8,
        )
        self.assertTrue(os.path.exists(checkpoint_step_4))
        self.assertTrue(os.path.exists(checkpoint_step_8))

        ray.init(ignore_reinit_error=True)
        # test bench mode
        self.config.mode = "bench"
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.eval_on_latest_checkpoint = False
        self.config.check_and_update()
        bench(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        countdown_metrics = parser.metric_list("eval/countdown")
        copy_countdown_metrics = parser.metric_list("eval/copy_countdown")
        self.assertTrue(len(countdown_metrics) > 0)
        self.assertTrue(len(copy_countdown_metrics) > 0)
        countdown_metric_steps = parser.metric_steps(countdown_metrics[0])
        countdown_copy_metric_steps = parser.metric_steps(copy_countdown_metrics[0])
        self.assertEqual(2, len(countdown_metric_steps))
        self.assertEqual(2, len(countdown_copy_metric_steps))
        self.assertTrue(4 in countdown_metric_steps)
        self.assertTrue(8 in countdown_metric_steps)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerGSM8K(BaseTrainerCase):
    def test_trainer(self):
        """Test GSM8K."""
        # test both mode
        self.config.algorithm.algorithm_type = AlgorithmType.GRPO
        self.config.algorithm.repeat_times = 8
        self.config.algorithm.advantage_fn_type = "grpo_adv_fn"
        self.config.algorithm.advantage_fn_args = {}
        # self.config.buffer.batch_size = 96  # TODO: used for real testing
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.total_training_steps = 4
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        self.config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr = 1e-5
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        # TODO: used for real testing
        # rewards = parser.metric_values("critic/rewards/mean")
        # self.assertTrue(0.4 < rewards[0] < 0.55)
        # self.assertTrue(0.4 < rewards[1] < 0.55)
        # self.assertTrue(0.6 < rewards[2] < 0.7)
        # self.assertTrue(0.6 < rewards[3] < 0.7)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerGSM8KWithSFT(BaseTrainerCase):
    def test_trainer(self):
        """Test GSM8K With SFT."""
        # test both mode
        self.config.algorithm.algorithm_type = AlgorithmType.GRPO
        self.config.algorithm.repeat_times = 8
        self.config.algorithm.advantage_fn_type = "grpo_adv_fn"
        self.config.algorithm.advantage_fn_args = {}
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.buffer.trainer_input.sft_warmup_steps = 2
        self.config.buffer.trainer_input.sft_warmup_dataset = get_unittest_dataset_config(
            "sft_for_gsm8k"
        )
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.total_training_steps = 4
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        self.config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr = 1e-5
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)  # SFT
        self.assertEqual(parser.metric_max_step(actor_metrics[-1]), 4)  # RFT
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerDPO(BaseTrainerCase):
    def test_trainer(self):
        """Test DPO."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = AlgorithmType.DPO
        self.config.algorithm.policy_loss_fn = "dpo"
        self.config.algorithm.policy_loss_fn_args = {}
        # self.config.buffer.batch_size = 32
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config("dpo")
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.total_training_steps = 4
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        self.config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr = 5e-7
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)
