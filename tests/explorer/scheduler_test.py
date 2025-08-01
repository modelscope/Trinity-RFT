import asyncio
import time
import unittest
from typing import List

import ray
import torch

from tests.tools import get_template_config
from trinity.buffer.reader.queue_reader import QueueReader
from trinity.common.config import GenerationConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel
from trinity.common.workflows import Task
from trinity.common.workflows.workflow import WORKFLOWS, Workflow
from trinity.explorer.scheduler import Scheduler


@WORKFLOWS.register_module("dummy_workflow")
class DummyWorkflow(Workflow):
    def __init__(self, *, task, model, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.error_type = task.raw_task.get("error_type", "")
        self.seconds = None
        self.repeat_times = task.rollout_args.n
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
                tokens=torch.zeros(5),
                prompt_length=2,
                prompt_text=self.error_type or "success",
                info={"repeat_times": self.repeat_times},
            )
            for _ in range(self.repeat_times)
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
    ) -> None:
        pass

    def has_api_server(self) -> bool:
        return True

    def api_server_ready(self) -> str:
        return "http://localhosts:12345"


def generate_tasks(
    total_num: int,
    timeout_num: int = 0,
    exception_num: int = 0,
    timeout_seconds: int = 10,
    repeat_times: int = 1,
):
    """Generate some tasks for testing

    Args:
        total_num: number of normal tasks
        timeout_num: number of timeout tasks
        exception_num: number of exception tasks
        timeout_seconds: the timeout for timeout tasks
    """
    tasks = [
        Task(workflow=DummyWorkflow, raw_task={}, rollout_args=GenerationConfig(n=repeat_times))
        for _ in range(total_num)
    ]

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
        self.config.buffer.trainer_input.experience_buffer.max_read_timeout = 1
        self.config.algorithm.repeat_times = 1
        self.config.check_and_update()
        self.queue = QueueReader(
            self.config.buffer.trainer_input.experience_buffer, self.config.buffer
        )

    async def test_get_results(self):
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        tasks = generate_tasks(8)
        scheduler.schedule(tasks, batch_id=0)

        statuses, exps = await scheduler.get_results(batch_id=0, min_num=8, timeout=20)
        self.assertEqual(len(statuses), 8)
        self.assertEqual(len(exps), 0)
        self.assertEqual(len(self.queue.read(batch_size=8)), 8)
        with self.assertRaises(TimeoutError):
            self.queue.read(batch_size=1)

        for result in statuses:
            self.assertTrue(result.ok)

        for batch_id in range(1, 4):
            tasks = generate_tasks(4)
            scheduler.schedule(tasks, batch_id=batch_id)

        for batch_id in range(1, 4):
            self.assertTrue(scheduler.has_step(batch_id))
            statuses, exps = await scheduler.get_results(batch_id=batch_id, min_num=4, timeout=10)
            self.assertEqual(len(statuses), 4)
            self.assertEqual(len(exps), 0)
            self.assertFalse(scheduler.has_step(batch_id))
            self.assertEqual(len(self.queue.read(batch_size=4)), 4)
        with self.assertRaises(TimeoutError):
            self.queue.read(batch_size=1)

        tasks = generate_tasks(3)
        scheduler.schedule(tasks, batch_id=4)
        self.assertTrue(scheduler.has_step(4))
        statuses, exps = await scheduler.get_results(batch_id=4)
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(exps), 0)
        self.assertFalse(scheduler.has_step(4))
        self.assertEqual(len(self.queue.read(batch_size=3)), 3)

        # test timeout
        tasks = generate_tasks(2, timeout_num=2, timeout_seconds=10)
        scheduler.schedule(tasks, batch_id=0)

        start_time = time.time()
        statuses, exps = await scheduler.get_results(batch_id=0, min_num=4, timeout=3)
        end_time = time.time()

        self.assertLessEqual(end_time - start_time, 5)
        self.assertEqual(len(statuses), 2)
        self.assertEqual(len(self.queue.read(batch_size=2)), 2)

        # test run tasks after timeout
        tasks = generate_tasks(4)
        scheduler.schedule(tasks, batch_id=0)

        # actor restart is slow, set a big timeout
        statuses, exps = await scheduler.get_results(batch_id=0, timeout=20)
        self.assertEqual(len(statuses), 4)

        success_count = sum(1 for r in statuses if r.ok)
        self.assertEqual(success_count, 4)
        self.assertEqual(len(self.queue.read(batch_size=4)), 4)
        with self.assertRaises(TimeoutError):
            self.queue.read(batch_size=1)

        # test exception tasks
        tasks = generate_tasks(1, exception_num=3)
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await scheduler.get_results(batch_id=1, timeout=5)
        self.assertEqual(len(statuses), 4)

        success_count = sum(1 for r in statuses if r.ok)
        self.assertEqual(success_count, 1)
        self.assertEqual(len(self.queue.read(batch_size=1)), 1)
        with self.assertRaises(TimeoutError):
            self.queue.read(batch_size=1)

        # test clear_timeout_tasks
        tasks = generate_tasks(3, timeout_num=1, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=2)
        statuses, exps = await scheduler.get_results(
            batch_id=2, timeout=2, clear_timeout_tasks=False
        )
        self.assertEqual(len(statuses), 3)
        self.assertEqual(len(self.queue.read(batch_size=3)), 3)
        statuses, exps = await scheduler.get_results(
            batch_id=2, timeout=2, clear_timeout_tasks=False
        )
        self.assertEqual(len(statuses), 1)
        self.assertEqual(len(exps), 0)
        self.assertEqual(len(self.queue.read(batch_size=1)), 1)
        with self.assertRaises(TimeoutError):
            self.queue.read(batch_size=1)

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

        status0, exps0 = await scheduler.get_results(batch_id=0, min_num=4, timeout=1)
        status1, exps1 = await scheduler.get_results(batch_id=1, min_num=3, timeout=1)
        self.assertEqual(len(status0), 4)
        self.assertEqual(len(status1), 3)

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

    async def test_wait_all_timeout_with_multi_batch(self):
        self.config.explorer.max_timeout = 5
        self.config.explorer.rollout_model.engine_num = 4
        self.config.explorer.runner_per_model = 1

        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()

        tasks = generate_tasks(1, timeout_num=3, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=0)
        tasks = generate_tasks(2, timeout_num=2, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=1)
        tasks = generate_tasks(3, timeout_num=1, timeout_seconds=3)
        scheduler.schedule(tasks, batch_id=2)
        start_time = time.time()
        await scheduler.wait_all()
        end_time = time.time()
        self.assertTrue(
            end_time - start_time > 9,
            f"wait time should be greater than 9, but got {end_time - start_time}",
        )

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

        self.assertEqual(len(results[0][0]), 3)
        self.assertEqual(len(results[1][0]), 4)
        self.assertEqual(len(results[2][0]), 2)

        await scheduler.stop()

    async def test_scheduler_restart_after_stop(self):
        scheduler = Scheduler(self.config, [DummyModel.remote()])

        await scheduler.start()
        tasks = generate_tasks(2)
        scheduler.schedule(tasks, batch_id=0)
        results, exps = await scheduler.get_results(batch_id=0, min_num=2, timeout=10)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(exps), 0)
        await scheduler.stop()

        self.config.explorer.collect_experiences = True
        await scheduler.start()
        tasks = generate_tasks(3, repeat_times=2)
        scheduler.schedule(tasks, batch_id=1)
        results, exps = await scheduler.get_results(batch_id=1, min_num=3, timeout=10)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(exps), 3 * 2)
        await scheduler.stop()

    async def test_scheduler_all_methods(self):
        self.config.explorer.collect_experiences = True
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = generate_tasks(8)
        scheduler.schedule(tasks, batch_id=0)
        self.assertTrue(scheduler.has_step(0))
        statuses, exps = await scheduler.get_results(batch_id=0, min_num=8, timeout=20)
        self.assertEqual(len(statuses), 8)
        self.assertEqual(len(exps), 8)
        scheduler.schedule(tasks, batch_id=1)
        scheduler.schedule(tasks[:4], batch_id=2)
        self.assertFalse(scheduler.has_step(0))
        statuses, exps = await scheduler.get_results(batch_id=0, min_num=8)
        self.assertFalse(scheduler.has_step(0))
        self.assertEqual(len(statuses), 0)  # batch_id 0 has no more tasks
        self.assertEqual(len(exps), 0)
        self.assertFalse(scheduler.has_step(0))
        self.assertTrue(scheduler.has_step(1))
        self.assertTrue(scheduler.has_step(2))
        await scheduler.wait_all()
        st = time.time()
        statuses, exps = await scheduler.get_results(batch_id=1)
        et = time.time()
        self.assertTrue(et - st < 1.0)
        self.assertEqual(len(statuses), 8)
        self.assertEqual(len(exps), 8)
        self.assertFalse(scheduler.has_step(1))
        self.assertTrue(scheduler.has_step(2))
        st = time.time()
        statuses, exps = await scheduler.get_results(batch_id=2)
        et = time.time()
        self.assertTrue(et - st < 1.0)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(exps), 4)
        self.assertFalse(scheduler.has_step(2))
        await scheduler.stop()

    async def test_split_tasks(self):
        self.config.explorer.max_repeat_times_per_runner = 2
        self.config.check_and_update()
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        exp_list = []

        tasks = generate_tasks(4, repeat_times=8)  # ceil(8 / 2) == 4
        scheduler.schedule(tasks, batch_id=1)
        statuses, exps = await scheduler.get_results(batch_id=1)
        self.assertEqual(len(statuses), 4 * 4)
        exps = self.queue.read(batch_size=4 * 8)
        self.assertEqual(len(exps), 4 * 8)
        exp_list.extend(exps)
        with self.assertRaises(TimeoutError):
            self.queue.read(batch_size=1)

        tasks = generate_tasks(4, repeat_times=5)  # ceil(5 / 2) == 3
        scheduler.schedule(tasks, batch_id=2)
        statuses, exps = await scheduler.get_results(batch_id=2)
        self.assertEqual(len(statuses), 4 * 3)
        exps = self.queue.read(batch_size=4 * 5)
        self.assertEqual(len(exps), 4 * 5)
        exp_list.extend(exps)
        with self.assertRaises(TimeoutError):
            self.queue.read(batch_size=1)

        tasks = generate_tasks(3, repeat_times=1)  # ceil(1 / 2) == 1
        scheduler.schedule(tasks, batch_id=3)
        statuses, exps = await scheduler.get_results(batch_id=3)
        self.assertEqual(len(statuses), 3 * 1)
        exps = self.queue.read(batch_size=3 * 1)
        self.assertEqual(len(exps), 3 * 1)
        exp_list.extend(exps)
        with self.assertRaises(TimeoutError):
            self.queue.read(batch_size=1)

        # test task_id and unique_id
        group_ids = [exp.eid.tid for exp in exp_list]
        self.assertEqual(len(set(group_ids)), 11)  # 4 + 4 + 3
        unique_ids = [exp.eid.uid for exp in exp_list]
        self.assertEqual(len(unique_ids), len(set(unique_ids)))

        await scheduler.stop()

    async def test_multi_step_execution(self):
        self.config.explorer.max_repeat_times_per_runner = 1
        self.config.check_and_update()
        scheduler = Scheduler(self.config, [DummyModel.remote(), DummyModel.remote()])
        await scheduler.start()
        tasks = generate_tasks(2, repeat_times=4)

        n_steps = 3
        for i in range(1, n_steps + 1):
            scheduler.schedule(tasks, batch_id=i)
            statuses, exps = await scheduler.get_results(batch_id=i)
            self.assertEqual(len(statuses), 2 * 4)
            exps = self.queue.read(batch_size=2 * 4)
            self.assertEqual(len(exps), 2 * 4)

        await scheduler.stop()

    def tearDown(self):
        try:
            ray.shutdown()
        except Exception:
            pass
