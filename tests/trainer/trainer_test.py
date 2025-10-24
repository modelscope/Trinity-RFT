"""Tests for trainer."""

import multiprocessing
import os
import shutil
import time
import unittest
from copy import deepcopy
from datetime import datetime
from unittest import mock

import ray
from parameterized import parameterized_class

from tests.tools import (
    RayUnittestBase,
    TensorBoardParser,
    get_checkpoint_path,
    get_lora_config,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
    get_vision_language_model_path,
)
from trinity.cli.launcher import bench, both, explore, run, train
from trinity.common.config import (
    AlgorithmConfig,
    BufferConfig,
    Config,
    ExperienceBufferConfig,
    ExplorerInput,
    StageConfig,
    TaskSelectorConfig,
    TrainerInput,
)
from trinity.common.constants import (
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    StorageType,
    SyncMethod,
    SyncStyle,
)
from trinity.common.models.utils import get_checkpoint_dir_with_step_num
from trinity.manager.state_manager import StateManager


class BaseTrainerCase(RayUnittestBase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.buffer.total_epochs = 2
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.algorithm.repeat_times = 3
        self.config.project = "Trainer-unittest"
        self.config.name = f"trainer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.monitor.monitor_type = "tensorboard"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 2
        self.config.synchronizer.sync_method = SyncMethod.NCCL
        self.config.explorer.eval_interval = 4


@parameterized_class(
    ("strategy",),
    [
        ("fsdp",),
        ("megatron",),
    ],
)
class TestTrainerCountdown(BaseTrainerCase):
    def test_trainer(self):
        """Test the both and bench mode."""
        # test both mode
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.taskset.task_selector = TaskSelectorConfig(
            selector_type="shuffle", seed=42
        )
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("countdown", "test")
        )
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("copy_countdown", "test")
        )
        self.config.trainer.save_interval = 4
        self.config.check_and_update()
        _trainer_config = self.config.trainer.trainer_config
        if self.strategy == "megatron":
            _trainer_config.actor_rollout_ref.actor.strategy = "megatron"
            _trainer_config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = 2
            _trainer_config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = 2
            _trainer_config.critic.strategy = "megatron"
            _trainer_config.critic.megatron.tensor_model_parallel_size = 2
        _trainer_config.trainer.max_actor_ckpt_to_keep = 2
        _trainer_config.trainer.max_critic_ckpt_to_keep = 2
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
        actor_kl_metrics = parser.metric_list("actor/kl")
        self.assertTrue(len(actor_kl_metrics) > 0)
        critic_kl_metrics = parser.metric_list("critic/kl")
        self.assertTrue(len(critic_kl_metrics) > 0)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 8)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint
        checkpoint_step_4, _ = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
            step_num=4,
        )
        # check save lastest checkpoint
        checkpoint_step_8, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_step_4, "actor"))) > 0)
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_step_8, "actor"))) > 0)
        self.assertEqual(step_num, 8)
        ray.init(ignore_reinit_error=True, namespace=self.config.ray_namespace)
        # test bench mode
        self.config.mode = "bench"
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.bench_on_latest_checkpoint = False
        self.config.check_and_update()
        bench(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        for prefix in ["eval", "bench"]:
            countdown_metrics = parser.metric_list(f"{prefix}/countdown")
            copy_countdown_metrics = parser.metric_list(f"{prefix}/copy_countdown")
            self.assertTrue(len(countdown_metrics) > 0)
            self.assertTrue(len(copy_countdown_metrics) > 0)
            countdown_metric_steps = parser.metric_steps(countdown_metrics[0])
            countdown_copy_metric_steps = parser.metric_steps(copy_countdown_metrics[0])
            self.assertEqual([0, 4, 8], countdown_metric_steps)
            self.assertEqual([0, 4, 8], countdown_copy_metric_steps)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestStepAheadAsyncRL(BaseTrainerCase):
    def test_trainer(self):
        """Test the explore step ahead trainer."""
        # train 4 step, sync_offset=1, sync_interval=2
        # Explorer:
        # | 1 | 2 | 3 |sync| 4 |
        # |---|---|---|sync|---|
        # Trainer:
        #     | 1 | 2 |sync| 3 | 4 |
        #     |---|---|sync|---|---|
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.trainer.save_interval = 4
        self.config.synchronizer.sync_interval = 2
        self.config.synchronizer.sync_offset = 1
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 1
        self.config.trainer.trainer_config.trainer.max_critic_ckpt_to_keep = 1

        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)
        actor_kl_metrics = parser.metric_list("actor/kl")
        self.assertTrue(len(actor_kl_metrics) > 0)
        critic_kl_metrics = parser.metric_list("critic/kl")
        self.assertTrue(len(critic_kl_metrics) > 0)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint

        checkpoint_step_4, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertEqual(step_num, 4)
        self.assertTrue(os.path.exists(checkpoint_step_4))

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


@parameterized_class(
    ("fsdp_strategy", "offloading"),
    [
        ("fsdp", False),
        ("fsdp2", False),
        ("fsdp", True),
        ("fsdp2", True),
    ],
)
class TestTrainerGSM8K(BaseTrainerCase):
    def test_trainer(self):
        """Test GSM8K."""
        # test both mode
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.repeat_times = 4
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "epsilon": 1e-6,
        }
        # self.config.algorithm.repeat_times = 8  # TODO: used for real testing
        # self.config.buffer.batch_size = 96  # TODO: used for real testing
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        actor_rollout_ref = self.config.trainer.trainer_config.actor_rollout_ref
        actor_rollout_ref.actor.strategy = self.fsdp_strategy
        actor_rollout_ref.actor.optim.lr = 1e-5
        if self.fsdp_strategy == "fsdp":
            actor_rollout_ref.actor.fsdp_config.param_offload = self.offloading
            actor_rollout_ref.actor.fsdp_config.optimizer_offload = self.offloading
            actor_rollout_ref.ref.fsdp_config.param_offload = self.offloading
            actor_rollout_ref.ref.fsdp_config.optimizer_offload = self.offloading
        else:  # fsdp2
            actor_rollout_ref.actor.fsdp_config.offload_policy = self.offloading
            actor_rollout_ref.ref.fsdp_config.offload_policy = self.offloading
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        pipeline_metrics = parser.metric_list("pipeline")
        self.assertTrue(len(pipeline_metrics) > 0)
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

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerSFTWarmupGSM8K(BaseTrainerCase):
    @mock.patch("trinity.cli.launcher.load_config")
    def test_trainer(self, mock_load):
        """Test GSM8K With SFT."""
        # test both mode
        self.config.synchronizer.sync_interval = 1
        self.config.trainer.save_interval = 8
        self.config.stages = [
            StageConfig(
                stage_name="sft_warmup",
                mode="train",
                algorithm=AlgorithmConfig(algorithm_type="sft"),
                buffer=BufferConfig(
                    total_steps=3,
                    train_batch_size=4,
                    trainer_input=TrainerInput(
                        experience_buffer=get_unittest_dataset_config("sft_for_gsm8k")
                    ),
                ),
            ),
            StageConfig(
                stage_name="grpo",
                mode="both",
                algorithm=AlgorithmConfig(
                    algorithm_type="grpo",
                    repeat_times=4,
                ),
                buffer=BufferConfig(
                    batch_size=4,
                    explorer_input=ExplorerInput(taskset=get_unittest_dataset_config("gsm8k")),
                    trainer_input=TrainerInput(
                        experience_buffer=ExperienceBufferConfig(
                            name="test_queue_storage",
                            max_read_timeout=20,
                            storage_type=StorageType.QUEUE,
                            max_retry_times=10,
                        )
                    ),
                    total_epochs=1,
                ),
            ),
        ]
        self.config.check_and_update()

        mock_load.return_value = self.config

        run(config_path="dummy.yaml")

        stage_configs = [cfg.check_and_update() for cfg in self.config]

        # sft warmup stage
        sft_config = stage_configs[0]
        parser = TensorBoardParser(os.path.join(sft_config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) == 0)
        sft_metrics = parser.metric_list("actor/sft")
        self.assertTrue(len(sft_metrics) > 0)
        self.assertEqual(parser.metric_max_step(sft_metrics[0]), 3)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_min_step(response_metrics[0]), 1)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 3)

        # grpo stage
        grpo_config = stage_configs[1]
        parser = TensorBoardParser(os.path.join(grpo_config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        sft_metrics = parser.metric_list("actor/sft")
        self.assertTrue(len(sft_metrics) == 0)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_min_step(response_metrics[0]), 1)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        # test save checkpoint when sft finish
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=sft_config.checkpoint_job_dir, trainer_type="verl", step_num=2
            )[1],
            2,
        )
        # test save checkpoint at last step
        checkpoint_dir, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=grpo_config.checkpoint_job_dir,
            trainer_type="verl",
        )
        self.assertEqual(step_num, 4)
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_dir, "actor"))) > 0)

    def tearDown(self):
        # TODO: remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerDPO(BaseTrainerCase):
    def test_trainer(self):
        """Test DPO."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "dpo"
        self.config.algorithm.policy_loss_fn = "dpo"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.buffer.total_epochs = 2
        self.config.buffer.total_steps = 4  # step has higher priority than epoch
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 8
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config("dpo")
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        self.config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr = 5e-7
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerSFT(BaseTrainerCase):
    def test_trainer(self):
        """Test SFT."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "sft"
        self.config.algorithm.policy_loss_fn = "sft"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.entropy_loss_fn = "none"
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 4
        self.config.buffer.total_epochs = 2
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config(
            "sft_for_gsm8k"
        )
        self.config.check_and_update()
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerToolsSFT(BaseTrainerCase):
    def test_trainer_tools(self):
        """Test SFT with tools."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "sft"
        self.config.algorithm.policy_loss_fn = "sft"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.entropy_loss_fn = "none"
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 4
        self.config.buffer.total_epochs = 4
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config(
            "sft_with_tools"
        )
        self.config.check_and_update()
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


def run_trainer(config: Config) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    train(config)


def run_explorer(config: Config) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    explore(config)


def run_both(config: Config) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    both(config)


@parameterized_class(
    ("use_priority_queue", "strategy"),
    [(False, "fsdp"), (True, "fsdp"), (True, "megatron")],
)
class TestFullyAsyncMode(unittest.TestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

    def test_fully_async_mode(self):
        config = get_template_config()
        config.project = "unittest"
        config.name = f"fully_async_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
        config.buffer.trainer_input.experience_buffer.replay_buffer.enable = self.use_priority_queue
        config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        config.synchronizer.sync_style = SyncStyle.DYNAMIC_BY_EXPLORER
        config.synchronizer.sync_interval = 8
        config.monitor.monitor_type = "tensorboard"
        trainer_config = deepcopy(config)
        trainer_config.mode = "train"
        trainer_config.buffer.train_batch_size = 4
        trainer_config.check_and_update()
        if self.strategy == "megatron":
            _trainer_config = trainer_config.trainer.trainer_config
            _trainer_config.actor_rollout_ref.actor.strategy = "megatron"
            _trainer_config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = 2
            _trainer_config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = 2
            _trainer_config.critic.strategy = "megatron"
            _trainer_config.critic.megatron.tensor_model_parallel_size = 2

        explorer1_config = deepcopy(config)
        explorer1_config.trainer = deepcopy(trainer_config.trainer)
        explorer1_config.mode = "explore"
        explorer1_config.explorer.name = "explorer1"
        config.cluster.gpu_per_node = 1
        config.cluster.node_num = 1
        explorer1_config.explorer.rollout_model.engine_num = 1
        explorer1_config.explorer.rollout_model.tensor_parallel_size = 1
        explorer1_config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE,
        )
        explorer2_config = deepcopy(explorer1_config)
        explorer2_config.trainer = deepcopy(trainer_config.trainer)
        explorer1_config.check_and_update()

        trainer_process = multiprocessing.Process(target=run_trainer, args=(trainer_config,))
        trainer_process.start()

        ray.init(ignore_reinit_error=True)
        while True:
            try:
                ray.get_actor("queue-exp_buffer", namespace=trainer_config.ray_namespace)
                break
            except ValueError:
                print("waiting for trainer to start.")
                time.sleep(5)

        explorer_process_1 = multiprocessing.Process(target=run_explorer, args=(explorer1_config,))
        explorer_process_1.start()

        time.sleep(5)
        explorer2_config.explorer.name = "explorer2"
        explorer2_config.check_and_update()
        explorer_process_2 = multiprocessing.Process(target=run_explorer, args=(explorer2_config,))
        explorer_process_2.start()

        explorer_process_1.join()
        explorer_process_2.join()

        # wait for trainer process to finish.
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
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        parser = TensorBoardParser(
            os.path.join(explorer2_config.monitor.cache_dir, "tensorboard", "explorer2")
        )
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        # check the checkpoint
        explorer1_cache = StateManager(
            path=explorer1_config.checkpoint_job_dir,
            trainer_name=None,
            explorer_name="explorer1",
            config=explorer1_config,
        )
        cache = explorer1_cache.load_explorer()
        self.assertEqual(cache["latest_iteration"], 4)
        explorer2_cache = StateManager(
            path=explorer2_config.checkpoint_job_dir,
            trainer_name=None,
            explorer_name="explorer2",
            config=explorer2_config,
        )
        cache = explorer2_cache.load_explorer()
        self.assertEqual(cache["latest_iteration"], 4)
        trainer_cache = StateManager(
            path=trainer_config.checkpoint_job_dir,
            trainer_name=trainer_config.trainer.name,
            config=trainer_config,
        )
        cache = trainer_cache.load_trainer()
        self.assertEqual(cache["latest_iteration"], 8)
        # check the lastest checkpoint
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=explorer1_config.checkpoint_job_dir,
                trainer_type="verl",
            )[1],
            8,
        )
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=explorer2_config.checkpoint_job_dir,
                trainer_type="verl",
            )[1],
            8,
        )
        log_files = os.listdir(os.path.join(explorer1_config.checkpoint_job_dir, "log"))
        self.assertTrue("trainer.log" in log_files)
        self.assertTrue("synchronizer.log" in log_files)
        self.assertTrue("explorer1.log" in log_files)
        self.assertTrue("explorer2.log" in log_files)
        self.assertTrue("explorer1_runner_0.log" in log_files)
        self.assertTrue("explorer1_runner_7.log" in log_files)
        self.assertTrue("explorer2_runner_0.log" in log_files)
        self.assertTrue("explorer2_runner_7.log" in log_files)
        self.assertTrue("explorer1_experience_pipeline.log" in log_files)
        self.assertTrue("explorer2_experience_pipeline.log" in log_files)
        files_to_check = ["trainer.log", "synchronizer.log", "explorer1.log", "explorer2.log"]
        for file_name in files_to_check:
            with open(os.path.join(explorer1_config.checkpoint_job_dir, "log", file_name)) as f:
                lines = f.readlines()
                self.assertTrue(len(lines) > 0)
        ray.shutdown()

    def tearDown(self):
        checkpoint_path = get_checkpoint_path()
        shutil.rmtree(os.path.join(checkpoint_path, "unittest"))


@parameterized_class(
    ("strategy",),
    [
        ("fsdp",),
        ("megatron",),
    ],
)
class TestTrainerCheckpointSave(unittest.TestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        self.config = get_template_config()
        self.config.buffer.total_epochs = 1
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.algorithm.repeat_times = 3
        self.config.project = "Trainer-unittest"
        self.config.name = f"trainer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.monitor.monitor_type = "tensorboard"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 1
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.eval_interval = 4
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.trainer.save_interval = 4
        self.config.check_and_update()

    def test_trainer(self):
        """Test the checkpoint saving."""
        _trainer_config = self.config.trainer.trainer_config
        if self.strategy == "megatron":
            _trainer_config.actor_rollout_ref.actor.strategy = "megatron"
            _trainer_config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = 2
            _trainer_config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = 2
            _trainer_config.critic.strategy = "megatron"
            _trainer_config.critic.megatron.tensor_model_parallel_size = 2
        _trainer_config.trainer.max_actor_ckpt_to_keep = 2
        _trainer_config.trainer.max_critic_ckpt_to_keep = 2

        trainer_process = multiprocessing.Process(target=run_both, args=(self.config,))
        trainer_process.start()

        default_local_dir = _trainer_config.trainer.default_local_dir
        state_dict_iteration = checkpoint_iteration = 0
        state_dict_iteration_file = os.path.join(
            default_local_dir, "latest_state_dict_iteration.txt"
        )
        checkpoint_iteration_file = os.path.join(
            default_local_dir, "latest_checkpointed_iteration.txt"
        )

        megatron_dist_ckpt_items = {
            "__0_1.distcp",
            "__1_0.distcp",
            "common.pt",
            ".metadata",
            "metadata.json",
            "__1_1.distcp",
            "__0_0.distcp",
        }
        while state_dict_iteration < 4 and checkpoint_iteration < 4:
            if os.path.exists(state_dict_iteration_file):
                try:
                    with open(state_dict_iteration_file, "r") as f:
                        state_dict_iteration = int(f.read().strip())
                except (IOError, ValueError):
                    pass
            if os.path.exists(checkpoint_iteration_file):
                try:
                    with open(checkpoint_iteration_file, "r") as f:
                        checkpoint_iteration = int(f.read().strip())
                except (IOError, ValueError):
                    pass

            if state_dict_iteration > 0:
                iteration_dir = os.path.join(
                    default_local_dir, f"global_step_{state_dict_iteration}", "actor"
                )
                if self.strategy == "fsdp":
                    items = os.listdir(iteration_dir)
                    self.assertIn("model_world_size_2_rank_0.pt", items)
                    self.assertIn("model_world_size_2_rank_1.pt", items)
                else:  # megatron
                    dist_ckpt_dir = os.path.join(iteration_dir, "dist_ckpt")
                    self.assertEqual(
                        set(os.listdir(dist_ckpt_dir)),
                        megatron_dist_ckpt_items,
                    )
                    huggingface_dir = os.path.join(iteration_dir, "huggingface")
                    items = os.listdir(huggingface_dir)
                    self.assertIn("config.json", items)
                    self.assertIn("generation_config.json", items)
                print(f"State dict check at {state_dict_iteration} iteration passed.")

            if checkpoint_iteration > 0:
                for sub_dir_name in ["actor", "critic"]:
                    iteration_dir = os.path.join(
                        default_local_dir, f"global_step_{checkpoint_iteration}", sub_dir_name
                    )
                    if self.strategy == "fsdp":
                        self.assertEqual(
                            set(os.listdir(iteration_dir)),
                            {
                                "model_world_size_2_rank_0.pt",
                                "model_world_size_2_rank_1.pt",
                                "optim_world_size_2_rank_1.pt",
                                "optim_world_size_2_rank_0.pt",
                                "extra_state_world_size_2_rank_0.pt",
                                "extra_state_world_size_2_rank_1.pt",
                                "huggingface",
                                "fsdp_config.json",
                            },
                        )
                    else:  # megatron
                        dist_ckpt_dir = os.path.join(iteration_dir, "dist_ckpt")
                        self.assertEqual(
                            set(os.listdir(dist_ckpt_dir)),
                            megatron_dist_ckpt_items,
                        )
                    huggingface_dir = os.path.join(iteration_dir, "huggingface")
                    self.assertEqual(
                        set(os.listdir(huggingface_dir)) - {"generation_config.json"},
                        {
                            "vocab.json",
                            "merges.txt",
                            "added_tokens.json",
                            "tokenizer.json",
                            "config.json",
                            "chat_template.jinja",
                            "tokenizer_config.json",
                            "special_tokens_map.json",
                        },
                    )
                print(f"Checkpoint check at {checkpoint_iteration} iteration passed.")

            time.sleep(1)
        trainer_process.join()

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerMIX(BaseTrainerCase):
    def test_trainer(self):
        """Test MIX algorithm."""
        # gsm8k has 16 tasks, sft_for_gsm8k has 8 tasks
        # total 4 steps, each step: read 4 tasks from gsm8k, 16 tasks from sft_for_gsm8k
        self.config.algorithm.algorithm_type = "mix"
        self.config.algorithm.repeat_times = 4
        self.config.algorithm.sample_strategy = "mix"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.sample_strategy_args = {"expert_data_ratio": 0.5}  # rft=4*4 : sft=16
        self.config.algorithm.policy_loss_fn = "mix"
        self.config.buffer.batch_size = 4
        self.config.buffer.train_batch_size = 32
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.synchronizer.sync_interval = 1
        self.config.trainer.save_interval = 1
        self.config.buffer.trainer_input.auxiliary_buffers[
            "sft_dataset"
        ] = get_unittest_dataset_config("sft_for_gsm8k")
        self.config.buffer.trainer_input.auxiliary_buffers[
            "sft_dataset"
        ].total_epochs = 8  # test this works
        self.config.check_and_update()
        self.config.buffer.trainer_input.experience_buffer.max_read_timeout = 20
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))

        # test rollout metrics
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        self.assertEqual(
            parser.metric_values("pipeline/experience_count")[1], 16
        )  # 16 rft experiences
        # test actor metrics
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        expert_metrics = parser.metric_list("actor/expert/")
        self.assertEqual(parser.metric_max_step(expert_metrics[0]), 4)  # SFT
        usual_metrics = parser.metric_list("actor/usual/")
        self.assertEqual(parser.metric_max_step(usual_metrics[0]), 4)  # RFT
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_min_step(response_metrics[0]), 1)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        # test save checkpoint at last step
        checkpoint_dir, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type="verl",
        )
        self.assertEqual(step_num, 4)
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_dir, "actor"))) > 0)

    def tearDown(self):
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestMultiModalGRPO(BaseTrainerCase):
    @unittest.skip("Require specific vllm/transformers version")
    def test_trainer(self):
        """Test both mode with multi-modal data."""
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config(
            "geometry"
        )  # Total 8 tasks
        self.config.model.model_path = get_vision_language_model_path()
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.repeat_times = 4
        self.config.buffer.batch_size = 4
        self.config.buffer.total_epochs = 1
        self.config.trainer.save_interval = 2
        self.config.check_and_update()
        both(self.config)
        # check metrics are available
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 2)
        # check save lastest checkpoint
        checkpoint_step_2, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_step_2, "actor"))) > 0)
        self.assertEqual(step_num, 2)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestMultiModalSFT(BaseTrainerCase):
    @unittest.skip("Require specific vllm/transformers version")
    def test_trainer(self):
        """Test SFT mode with multi-modal data."""
        self.config.mode = "train"
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config(
            "geometry"
        )  # Total 8 tasks
        self.config.model.model_path = get_vision_language_model_path()
        self.config.algorithm.algorithm_type = "sft"
        self.config.algorithm.policy_loss_fn = "sft"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.entropy_loss_fn = "none"
        self.config.buffer.train_batch_size = 4
        self.config.buffer.total_epochs = 1
        self.config.trainer.save_interval = 2
        self.config.check_and_update()
        train(self.config)
        # check metrics are available
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 2)
        # check save lastest checkpoint
        checkpoint_step_2, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_step_2, "actor"))) > 0)
        self.assertEqual(step_num, 2)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerLoRA(BaseTrainerCase):
    def test_trainer(self):
        """Test both mode with LoRA request."""
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("gsm8k", "test")
        )
        self.config.model.model_path = get_model_path()
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.repeat_times = 4
        self.config.buffer.batch_size = 4
        self.config.buffer.total_steps = 2
        self.config.cluster.node_num = 1
        self.config.cluster.gpu_per_node = 4
        self.config.explorer.eval_interval = 2
        self.config.model.lora_configs = [get_lora_config()]
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.synchronizer.sync_interval = 2
        self.config.trainer.save_interval = 2
        self.config.check_and_update()
        both(self.config)
        # check metrics are available
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 2)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 2)
        ray.shutdown(_exiting_interpreter=True)
        # check save lastest checkpoint
        checkpoint_step_2, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_step_2, "actor"))) > 0)
        self.assertTrue(
            len(os.listdir(os.path.join(checkpoint_step_2, "actor", "lora_adapter"))) > 0
        )
        self.assertEqual(step_num, 2)

        # test bench mode
        ray.init(ignore_reinit_error=True, namespace=self.config.ray_namespace)
        self.config.mode = "bench"
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.bench_on_latest_checkpoint = False
        self.config.check_and_update()
        bench(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        for prefix in ["eval", "bench"]:
            gsm8k_metrics = parser.metric_list(f"{prefix}/gsm8k")
            self.assertTrue(len(gsm8k_metrics) > 0)
            gsm8k_metric_steps = parser.metric_steps(gsm8k_metrics[0])
            self.assertEqual([0, 2], gsm8k_metric_steps)

    def tearDown(self):
        shutil.rmtree(self.config.checkpoint_job_dir)
