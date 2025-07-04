import asyncio
import time
import unittest
from typing import List, Tuple

import ray
import torch

from tests.tools import get_template_config
from trinity.buffer.reader.queue_reader import QueueReader
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.workflows import Task
from trinity.common.workflows.workflow import WORKFLOWS, Workflow
from trinity.explorer.scheduler import Scheduler


@WORKFLOWS.register_module("dummy_workflow")
class DummyWorkflow(Workflow):
    def __init__(self, model, task, auxiliary_models):
        super().__init__(model, task, auxiliary_models)
        self.error_type = task.raw_task.get("error_type", "")
        self.seconds = None
        if "timeout" in self.error_type:
            parts = self.error_type.split("_")
            if len(parts) > 1:
                self.seconds = int(parts[-1])
            else:
                self.seconds = 10

    def run(self) -> List[Experience]:
        if "timeout" in self.error_type:
            time.sleep(self.seconds)
        elif self.error_type == "exception":
            raise ValueError("Exception occurred")
        elif self.error_type == "exit":
            exit(1)
        elif self.error_type == "auxiliary_models":
            assert self.auxiliary_models is not None and len(self.auxiliary_models) == 2

        return [
            Experience(
                tokens=torch.zeros(5), prompt_length=2, prompt_text=self.error_type or "success"
            )
        ]


@ray.remote
class DummyModel(InferenceModel):
    def sync_model(self, model_version, update_weight_args_list):
        return True

    def get_model_version(self):
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


@ray.remote
class DummyAuxiliaryModel(InferenceModel):
    def sync_model(self, model_version, update_weight_args_list):
        return True

    def get_model_version(self):
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

    def has_api_server(self) -> bool:
        return True

    def api_server_ready(self) -> Tuple[str, str]:
        return "http://localhosts:12345", "placeholder"


def generate_tasks(
    total_num: int, timeout_num: int = 0, exception_num: int = 0, timeout_seconds: int = 10
):
    """Generate some tasks for testing

    Args:
        total_num: number of normal tasks
        timeout_num: number of timeout tasks
        exception_num: number of exception tasks
        timeout_seconds: the timeout for timeout tasks
    """
    tasks = [Task(workflow=DummyWorkflow, raw_task={}) for _ in range(total_num)]

    tasks.extend(
        [
            Task(
                workflow=DummyWorkflow,
                raw_task={"error_type": f"timeout_{timeout_seconds}"},
            )
            for _ in range(timeout_num)
        ]
    )

    tasks.extend(
        [
            Task(
                workflow=DummyWorkflow,
                raw_task={"error_type": "exception"},
            )
            for _ in range(exception_num)
        ]
    )

    return tasks


class SchedulerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.explorer.max_retry_times = 1
        self.config.explorer.max_timeout = 5
        self.config.explorer.runner_per_model = 2
        self.config.buffer.read_batch_size = 2
        self.config.buffer.pad_token_id = 0
        self.config.buffer.explorer_output = (
            self.config.buffer.trainer_input.experience_buffer
        ) = StorageConfig(
            name="test",
            storage_type=StorageType.QUEUE,
            algorithm_type="ppo",
            path="",
        )
        self.queue = QueueReader(
            self.config.buffer.trainer_input.experience_buffer, self.config.buffer
        )

    async def test_get_results(self):
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        tasks = generate_tasks(8)
        scheduler.schedule(tasks, batch_id=0)

        results = await scheduler.get_results(batch_id=0, min_num=8, timeout=20)
        self.assertEqual(len(results), 8)

        for result in results:
            self.assertTrue(result.ok)

        for batch_id in range(1, 4):
            tasks = generate_tasks(4)
            scheduler.schedule(tasks, batch_id=batch_id)

        for batch_id in range(1, 4):
            self.assertTrue(scheduler.has_step(batch_id))
            results = await scheduler.get_results(batch_id=batch_id, min_num=4, timeout=10)
            self.assertEqual(len(results), 4)
            self.assertFalse(scheduler.has_step(batch_id))

        tasks = generate_tasks(3)
        scheduler.schedule(tasks, batch_id=4)
        self.assertTrue(scheduler.has_step(4))
        results = await scheduler.get_results(batch_id=4)
        self.assertEqual(len(results), 3)
        self.assertFalse(scheduler.has_step(4))

        # test timeout
        tasks = generate_tasks(2, timeout_num=2, timeout_seconds=10)
        scheduler.schedule(tasks, batch_id=0)

        start_time = time.time()
        results = await scheduler.get_results(batch_id=0, min_num=4, timeout=3)
        end_time = time.time()

        self.assertLessEqual(end_time - start_time, 5)
        self.assertEqual(len(results), 2)

        # test run tasks after timeout
        tasks = generate_tasks(4)
        scheduler.schedule(tasks, batch_id=0)

        # actor restart is slow, set a big timeout
        results = await scheduler.get_results(batch_id=0, timeout=20)
        self.assertEqual(len(results), 4)

        success_count = sum(1 for r in results if r.ok)

        self.assertEqual(success_count, sum(1 for r in results if r.ok))

        # test exception tasks
        tasks = generate_tasks(1, exception_num=3)
        scheduler.schedule(tasks, batch_id=1)
        results = await scheduler.get_results(batch_id=1, timeout=5)
        self.assertEqual(len(results), 4)

        success_count = sum(1 for r in results if r.ok)
        self.assertEqual(success_count, 1)

        # test clear_timeout_tasks
        tasks = generate_tasks(3, timeout_num=1, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=2)
        results = await scheduler.get_results(batch_id=2, timeout=2, clear_timeout_tasks=False)
        self.assertEqual(len(results), 3)
        results = await scheduler.get_results(batch_id=2, timeout=2, clear_timeout_tasks=False)
        self.assertEqual(len(results), 1)

        await scheduler.stop()

    async def test_wait_all(self):
        """Test wait all"""
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        tasks1 = generate_tasks(4)
        tasks2 = generate_tasks(3)
        scheduler.schedule(tasks1, batch_id=0)
        scheduler.schedule(tasks2, batch_id=1)

        start_time = time.time()
        await scheduler.wait_all(timeout=10.0)
        end_time = time.time()

        self.assertLess(end_time - start_time, 5.0)

        self.assertEqual(len(scheduler.pending_tasks), 0)
        self.assertEqual(len(scheduler.running_tasks), 0)

        results0 = await scheduler.get_results(batch_id=0, min_num=4, timeout=1)
        results1 = await scheduler.get_results(batch_id=1, min_num=3, timeout=1)
        self.assertEqual(len(results0), 4)
        self.assertEqual(len(results1), 3)

        # test timeout
        tasks = generate_tasks(2, timeout_num=2, timeout_seconds=10)
        scheduler.schedule(tasks, batch_id=0)

        start_time = time.time()
        with self.assertRaises(TimeoutError):
            await scheduler.wait_all(timeout=3.0)
        end_time = time.time()

        self.assertGreaterEqual(end_time - start_time, 2.8)
        self.assertLessEqual(end_time - start_time, 4.0)

        # test empty scenario

        start_time = time.time()
        await scheduler.wait_all(timeout=5.0)
        end_time = time.time()

        self.assertLess(end_time - start_time, 1.0)
        await scheduler.stop()

    async def test_concurrent_operations(self):
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        async def schedule_tasks(batch_id, num_tasks):
            tasks = generate_tasks(num_tasks)
            scheduler.schedule(tasks, batch_id=batch_id)
            return await scheduler.get_results(batch_id=batch_id, min_num=num_tasks, timeout=10)

        results = await asyncio.gather(
            schedule_tasks(0, 3),
            schedule_tasks(1, 4),
            schedule_tasks(2, 2),
        )

        self.assertEqual(len(results[0]), 3)
        self.assertEqual(len(results[1]), 4)
        self.assertEqual(len(results[2]), 2)

        await scheduler.stop()

    async def test_scheduler_restart_after_stop(self):
        scheduler = Scheduler(self.config, [DummyModel.remote()])

        await scheduler.start()
        tasks = generate_tasks(2)
        scheduler.schedule(tasks, batch_id=0)
        results = await scheduler.get_results(batch_id=0, min_num=2, timeout=10)
        self.assertEqual(len(results), 2)
        await scheduler.stop()

        await scheduler.start()
        tasks = generate_tasks(3)
        scheduler.schedule(tasks, batch_id=1)
        results = await scheduler.get_results(batch_id=1, min_num=3, timeout=10)
        self.assertEqual(len(results), 3)
        await scheduler.stop()

    async def test_scheduler_all_methods(self):
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = generate_tasks(8)
        scheduler.schedule(tasks, batch_id=0)
        self.assertTrue(scheduler.has_step(0))
        results = await scheduler.get_results(batch_id=0, min_num=8, timeout=20)
        self.assertEqual(len(results), 8)
        scheduler.schedule(tasks, batch_id=1)
        scheduler.schedule(tasks[:4], batch_id=2)
        self.assertFalse(scheduler.has_step(0))
        results = await scheduler.get_results(batch_id=0, min_num=8)
        self.assertFalse(scheduler.has_step(0))
        self.assertEqual(len(results), 0)  # batch_id 0 has no more tasks
        self.assertFalse(scheduler.has_step(0))
        self.assertTrue(scheduler.has_step(1))
        self.assertTrue(scheduler.has_step(2))
        await scheduler.wait_all()
        st = time.time()
        results = await scheduler.get_results(batch_id=1)
        et = time.time()
        self.assertTrue(et - st < 1.0)
        self.assertEqual(len(results), 8)
        self.assertFalse(scheduler.has_step(1))
        self.assertTrue(scheduler.has_step(2))
        st = time.time()
        results = await scheduler.get_results(batch_id=2)
        et = time.time()
        self.assertTrue(et - st < 1.0)
        self.assertEqual(len(results), 4)
        self.assertFalse(scheduler.has_step(2))
        await scheduler.stop()

    def tearDown(self):
        try:
            ray.shutdown()
        except Exception:
            pass
