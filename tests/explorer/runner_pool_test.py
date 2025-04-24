import os
import time
import unittest
from typing import List

import ray
import torch

from trinity.buffer.reader.queue_reader import QueueReader
from trinity.common.config import DatasetConfig, load_config
from trinity.common.constants import AlgorithmType, StorageType
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.task import Task
from trinity.common.workflows.workflow import WORKFLOWS, Workflow
from trinity.explorer.runner_pool import RunnerPool

config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data", "template.yaml")


@WORKFLOWS.register_module("dummy_workflow")
class DummyWorkflow(Workflow):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.error_type = kwargs.get("task_desc")
        self.seconds = None
        if "timeout" in self.error_type:
            self.seconds = int(self.error_type.split("_")[-1])

    def run(self) -> List[Experience]:
        if "timeout" in self.error_type:
            time.sleep(self.seconds)
        elif self.error_type == "exception":
            raise RuntimeError("Exception occurred")
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
        offline_update: bool = True,
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
        self.config.buffer.train_dataset = DatasetConfig(
            name="test",
            storage_type=StorageType.QUEUE,
            algorithm_type=AlgorithmType.PPO,
        )
        self.queue = QueueReader(self.config.buffer.train_dataset, self.config.buffer)

    def test_runner_pool(self):
        pool = RunnerPool(self.config, [DummyModel.remote(), DummyModel.remote()])
        tasks = [
            Task(
                task_desc="timeout_100",
                workflow=DummyWorkflow,
            ),
            Task(
                task_desc="exception",
                workflow=DummyWorkflow,
            ),
            Task(
                task_desc="timeout_2",
                workflow=DummyWorkflow,
            ),
            Task(
                task_desc="success",
                workflow=DummyWorkflow,
            ),
            Task(
                task_desc="timeout_101",
                workflow=DummyWorkflow,
            ),
            Task(
                task_desc="exit",
                workflow=DummyWorkflow,
            ),
        ]

        pool.run_tasks(
            tasks=tasks,
        )

        # The excepted return order is: `exception` -> `timeout_5` -> `success` -> (`timeout_100`and `timeout_101`) -> `exit`
        # 1. `exception`
        st = time.time()
        status = pool.get_next_unorder()
        et = time.time()
        self.assertTrue(et - st < 5)
        self.assertEqual(len(status), 1)
        self.assertFalse(status[0].ok)
        # 2. `timeout_2
        st = time.time()
        status = pool.get_next_unorder()
        et = time.time()
        self.assertTrue(et - st < 3)
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
