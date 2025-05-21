import os
import time
import unittest
from typing import List

import ray
import torch

from tests.tools import get_unittest_dataset_config
from trinity.buffer.reader.queue_reader import QueueReader
from trinity.common.config import StorageConfig, load_config
from trinity.common.constants import AlgorithmType, StorageType
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.workflows import Task
from trinity.common.workflows.workflow import WORKFLOWS, Workflow
from trinity.explorer.runner_pool import RunnerPool

config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data", "template.yaml")


@WORKFLOWS.register_module("dummy_workflow")
class DummyWorkflow(Workflow):
    def __init__(self, model, task, auxiliary_models):
        super().__init__(model, task)
        self.error_type = task.task_desc
        self.seconds = None
        if "timeout" in self.error_type:
            self.seconds = int(self.error_type.split("_")[-1])

    def run(self) -> List[Experience]:
        if "timeout" in self.error_type:
            time.sleep(self.seconds)
        elif self.error_type == "exception":
            raise ValueError("Exception occurred")
        elif self.error_type == "exit":
            exit(1)
        return [Experience(tokens=torch.zeros(5), prompt_length=2, prompt_text=self.error_type)]


@ray.remote
class DummyModel(InferenceModel):
    def sync_model(self, update_weight_args_list):
        return True

    def get_ckp_version(self):
        return 0

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        timeout: int = 1200,
        update_with_checkpoint: bool = True,
    ) -> None:
        pass


class RunnerPoolTest(unittest.TestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = load_config(config_dir)
        self.config.explorer.runner_num = 2
        self.config.explorer.max_retry_times = 0
        self.config.explorer.max_timeout = 5
        self.config.buffer.read_batch_size = 2
        self.config.buffer.pad_token_id = 0
        self.config.buffer.explorer_output = (
            self.config.buffer.trainer_input.experience_buffer
        ) = StorageConfig(
            name="test",
            storage_type=StorageType.QUEUE,
            algorithm_type=AlgorithmType.PPO,
        )
        self.queue = QueueReader(
            self.config.buffer.trainer_input.experience_buffer, self.config.buffer
        )

    def test_runner_pool(self):
        pool = RunnerPool(self.config, [DummyModel.remote(), DummyModel.remote()])
        taskset_config = get_unittest_dataset_config("countdown")
        tasks = [
            Task(
                workflow=DummyWorkflow,
                format_args=taskset_config.format,
                rollout_args=taskset_config.rollout_args,
                is_eval=False,
                raw_task={
                    taskset_config.format.prompt_key: "timeout_100",
                },
            ),
            Task(
                workflow=DummyWorkflow,
                format_args=taskset_config.format,
                rollout_args=taskset_config.rollout_args,
                is_eval=False,
                raw_task={
                    taskset_config.format.prompt_key: "exception",
                },
            ),
            Task(
                workflow=DummyWorkflow,
                format_args=taskset_config.format,
                rollout_args=taskset_config.rollout_args,
                is_eval=False,
                raw_task={
                    taskset_config.format.prompt_key: "timeout_2",
                },
            ),
            Task(
                workflow=DummyWorkflow,
                format_args=taskset_config.format,
                rollout_args=taskset_config.rollout_args,
                is_eval=False,
                raw_task={
                    taskset_config.format.prompt_key: "success",
                },
            ),
            Task(
                workflow=DummyWorkflow,
                format_args=taskset_config.format,
                rollout_args=taskset_config.rollout_args,
                is_eval=False,
                raw_task={
                    taskset_config.format.prompt_key: "timeout_101",
                },
            ),
            Task(
                workflow=DummyWorkflow,
                format_args=taskset_config.format,
                rollout_args=taskset_config.rollout_args,
                is_eval=False,
                raw_task={
                    taskset_config.format.prompt_key: "exit",
                },
            ),
        ]

        pool.run_tasks(
            tasks=tasks,
        )

        # The excepted return order is: `exception` -> `timeout_2` -> `success` -> (`timeout_100`and `timeout_101`) -> `exit`
        # 1. `exception`
        st = time.time()
        status = pool.get_next_unorder()
        et = time.time()
        self.assertTrue(et - st < 2)
        print(f"First task use time: {et - st}")
        self.assertEqual(len(status), 1)
        self.assertFalse(status[0].ok)
        # 2. `timeout_2
        st = time.time()
        status = pool.get_next_unorder()
        et = time.time()
        self.assertTrue(et - st > 2)
        self.assertEqual(len(status), 1)
        self.assertTrue(status[0].ok)
        # 3. `success`
        st = time.time()
        status = pool.get_next_unorder()
        et = time.time()
        self.assertTrue(et - st < 1)
        self.assertEqual(len(status), 1)
        self.assertTrue(status[0].ok)
        # 4. `timeout_100`and `timeout_101`
        st = time.time()
        status = pool.get_next_unorder()
        et = time.time()
        self.assertTrue(et - st > 5)
        self.assertEqual(len(status), 2)
        self.assertFalse(status[0].ok)
        self.assertFalse(status[1].ok)

        # 5.`exit`
        status = pool.get_next_unorder()
        self.assertEqual(len(status), 1)
        self.assertFalse(status[0].ok)

        exps = self.queue.read()
        self.assertEqual(len(exps), 2)  # `timeout_2` and `success`
        self.assertEqual(len(pool._idle_actors), self.config.explorer.runner_num)
