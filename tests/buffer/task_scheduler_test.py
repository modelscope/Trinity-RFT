import os
import shutil
import unittest
from typing import Dict, List

from parameterized import parameterized

from tests.tools import get_template_config
from trinity.buffer.task_scheduler import TasksetScheduler
from trinity.common.config import FormatConfig, TaskSelectorConfig, TasksetConfig
from trinity.common.workflows.workflow import Task


class TestTaskScheduler(unittest.IsolatedAsyncioTestCase):
    temp_output_path = "tmp/test_task_scheduler/"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.makedirs(cls.temp_output_path, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if os.path.exists(cls.temp_output_path):
            shutil.rmtree(cls.temp_output_path)

    def _check_batch_tasks(self, batch_tasks: List[Task], indices: List[Dict[str, int]]) -> None:
        for task, index in zip(batch_tasks, indices):
            self.assertEqual(task.index["taskset_id"], index["taskset_id"])
            self.assertEqual(task.index["index"], index["index"])
            self.assertEqual(
                task.raw_task["question"],  # type: ignore
                f"Question {index['index'] + 1} in subset {index['taskset_id'] + 1}.",
            )
            self.assertEqual(
                task.raw_task["answer"],  # type: ignore
                f"Answer {index['index'] + 1} in subset {index['taskset_id'] + 1}.",
            )

    @parameterized.expand(
        [
            (
                {"selector_type": "sequential"},
                [
                    {"index": 0, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 1, "taskset_id": 1},
                    {"index": 1, "taskset_id": 0},
                    {"index": 2, "taskset_id": 1},
                    {"index": 3, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 3, "taskset_id": 0},
                    {"index": 4, "taskset_id": 1},
                    {"index": 5, "taskset_id": 1},
                    {"index": 6, "taskset_id": 1},
                    {"index": 4, "taskset_id": 0},
                    {"index": 0, "taskset_id": 1},
                    {"index": 1, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 2, "taskset_id": 1},
                    {"index": 3, "taskset_id": 1},
                    {"index": 1, "taskset_id": 0},
                    {"index": 2, "taskset_id": 0},
                    {"index": 4, "taskset_id": 1},
                    {"index": 3, "taskset_id": 0},
                    {"index": 5, "taskset_id": 1},
                    {"index": 6, "taskset_id": 1},
                    {"index": 4, "taskset_id": 0},
                ],
            ),
            (
                {"selector_type": "shuffle", "seed": 42},
                [
                    {"index": 3, "taskset_id": 1},
                    {"index": 4, "taskset_id": 0},
                    {"index": 2, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 6, "taskset_id": 1},
                    {"index": 4, "taskset_id": 1},
                    {"index": 3, "taskset_id": 0},
                    {"index": 1, "taskset_id": 0},
                    {"index": 1, "taskset_id": 1},
                    {"index": 5, "taskset_id": 1},
                    {"index": 0, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 2, "taskset_id": 1},
                    {"index": 6, "taskset_id": 1},
                    {"index": 4, "taskset_id": 0},
                    {"index": 5, "taskset_id": 1},
                    {"index": 1, "taskset_id": 1},
                    {"index": 1, "taskset_id": 0},
                    {"index": 2, "taskset_id": 0},
                    {"index": 4, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 0, "taskset_id": 1},
                    {"index": 3, "taskset_id": 1},
                    {"index": 3, "taskset_id": 0},
                ],
            ),
            (
                {"selector_type": "random", "seed": 42},
                [
                    {"index": 0, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 3, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 4, "taskset_id": 1},
                    {"index": 0, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 0, "taskset_id": 0},
                    {"index": 6, "taskset_id": 1},
                    {"index": 3, "taskset_id": 1},
                    {"index": 0, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 0, "taskset_id": 1},
                    {"index": 2, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 2, "taskset_id": 1},
                    {"index": 6, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 0, "taskset_id": 0},
                    {"index": 5, "taskset_id": 1},
                    {"index": 3, "taskset_id": 0},
                    {"index": 2, "taskset_id": 1},
                    {"index": 6, "taskset_id": 1},
                    {"index": 1, "taskset_id": 0},
                ],
            ),
            (
                {"selector_type": "offline_easy2hard", "feature_keys": ["feature_offline"]},
                [
                    {"index": 3, "taskset_id": 1},
                    {"index": 3, "taskset_id": 0},
                    {"index": 4, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 1, "taskset_id": 1},
                    {"index": 0, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 4, "taskset_id": 0},
                    {"index": 6, "taskset_id": 1},
                    {"index": 5, "taskset_id": 1},
                    {"index": 2, "taskset_id": 1},
                    {"index": 1, "taskset_id": 0},
                    {"index": 3, "taskset_id": 1},
                    {"index": 4, "taskset_id": 1},
                    {"index": 3, "taskset_id": 0},
                    {"index": 1, "taskset_id": 1},
                    {"index": 0, "taskset_id": 1},
                    {"index": 0, "taskset_id": 0},
                    {"index": 2, "taskset_id": 0},
                    {"index": 6, "taskset_id": 1},
                    {"index": 4, "taskset_id": 0},
                    {"index": 5, "taskset_id": 1},
                    {"index": 2, "taskset_id": 1},
                    {"index": 1, "taskset_id": 0},
                ],
            ),
            (
                {"selector_type": "difficulty_based", "feature_keys": ["feat_1", "feat_2"]},
                [
                    {"index": 3, "taskset_id": 1},
                    {"index": 3, "taskset_id": 0},
                    {"index": 6, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 2, "taskset_id": 1},
                    {"index": 3, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 3, "taskset_id": 0},
                    {"index": 2, "taskset_id": 1},
                    {"index": 1, "taskset_id": 1},
                    {"index": 4, "taskset_id": 1},
                    {"index": 2, "taskset_id": 0},
                    {"index": 3, "taskset_id": 1},
                    {"index": 2, "taskset_id": 1},
                    {"index": 4, "taskset_id": 0},
                    {"index": 4, "taskset_id": 1},
                    {"index": 5, "taskset_id": 1},
                    {"index": 4, "taskset_id": 0},
                    {"index": 3, "taskset_id": 0},
                    {"index": 5, "taskset_id": 1},
                    {"index": 1, "taskset_id": 0},
                    {"index": 6, "taskset_id": 1},
                    {"index": 6, "taskset_id": 1},
                    {"index": 4, "taskset_id": 0},
                ],
            ),
        ]
    )
    async def test_task_scheduler(self, task_selector_kwargs, batch_tasks_orders) -> None:
        config = get_template_config()
        config.buffer.batch_size = 2
        config.buffer.total_epochs = 2
        config.buffer.explorer_input.taskset = None
        config.buffer.explorer_input.tasksets = [
            TasksetConfig(
                name="subset_1",
                path=os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "template",
                    "data",
                    "task_scheduler",
                    "subset_1",
                ),
                split="train",
                enable_progress_bar=False,
                format=FormatConfig(
                    prompt_key="question",
                    response_key="answer",
                ),
                default_workflow_type="math_workflow",
                default_reward_fn_type="math_reward",
                task_selector=TaskSelectorConfig(
                    **task_selector_kwargs,
                ),
            ),
            TasksetConfig(
                name="subset_2",
                path=os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "template",
                    "data",
                    "task_scheduler",
                    "subset_2",
                ),
                split="train",
                enable_progress_bar=False,
                format=FormatConfig(
                    prompt_key="question",
                    response_key="answer",
                ),
                default_workflow_type="math_workflow",
                default_reward_fn_type="math_reward",
                task_selector=TaskSelectorConfig(
                    **task_selector_kwargs,
                ),
            ),
        ]
        config.check_and_update()

        task_scheduler = TasksetScheduler({}, config)
        self.assertEqual(len(batch_tasks_orders) % config.buffer.batch_size, 0)
        for i, start_id in enumerate(range(0, len(batch_tasks_orders), config.buffer.batch_size)):
            batch_tasks_indices = batch_tasks_orders[start_id : start_id + config.buffer.batch_size]
            batch_tasks = await task_scheduler.read_async()
            # for task in batch_tasks:  # used for debug
            #     print(f"{task.index},")
            self._check_batch_tasks(batch_tasks, batch_tasks_indices)
            if i % 3 == 2:
                # test resume
                state_dict = {
                    "latest_iteration": task_scheduler.step,
                    "taskset_states": task_scheduler.state_dict(),
                }
                task_scheduler = TasksetScheduler(state_dict, config)

        with self.assertRaises(StopAsyncIteration):
            batch_tasks = await task_scheduler.read_async()
