"""Scheduler for rollout tasks."""

import asyncio
import re
import time
import traceback
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Union

import ray

from trinity.common.config import Config
from trinity.common.models import InferenceModel
from trinity.common.workflows import Task
from trinity.explorer.workflow_runner import Status, WorkflowRunner
from trinity.utils.log import get_logger


class RunnerWrapper:
    """A wrapper for a WorkflowRunner"""

    def __init__(
        self,
        runner_id: int,
        rollout_model: InferenceModel,
        auxiliary_models: List[InferenceModel],
        config: Config,
    ):
        self.logger = get_logger(__name__)
        self.runner_id = runner_id
        self.rollout_model = rollout_model
        self.auxiliary_models = auxiliary_models
        self.config = config
        self.retry_times = config.explorer.max_retry_times
        self.timeout = config.explorer.max_timeout
        self.namespace = ray.get_runtime_context().namespace
        self.runner = self._create_runner()

    def _create_runner(self):
        return (
            ray.remote(WorkflowRunner)
            .options(
                namespace=self.namespace,
                scheduling_strategy="SPREAD",
            )
            .remote(self.config, self.rollout_model, self.auxiliary_models)
        )

    async def run_with_retry(self, task: Task) -> Tuple[Status, int]:
        """
        Returns:
            `Status`: The return status of the task.
            `int`: The runner_id of current runner.
        """
        last_exception_msg = None
        await self.runner.__ray_ready__.remote()
        start_time = time.time()
        status = Status(ok=False, metric=dict())
        try:
            for attempt in range(self.retry_times + 1):
                try:
                    status = await asyncio.wait_for(self.runner.run_task.remote(task), self.timeout)
                    if status.ok:
                        break
                    else:
                        self.logger.error(status.message)
                except asyncio.TimeoutError:
                    last_exception_msg = (
                        f"Timeout when running task at runner {self.runner_id}: {task}"
                    )
                    self.logger.error(last_exception_msg)
                    status = Status(ok=False, metric=dict(), message=last_exception_msg)
                except Exception:
                    last_exception_msg = traceback.format_exc()
                    self.logger.warning(
                        f"Task execution attempt {attempt + 1} failed:\n{last_exception_msg}"
                    )
                    status = Status(ok=False, metric=dict(), message=last_exception_msg)
        finally:
            end_time = time.time()
            status.metric["task_run_time"] = end_time - start_time
        return status, self.runner_id

    def restart_runner(self):
        old_runner = self.runner
        self.runner = self._create_runner()
        try:
            ray.kill(old_runner)
        except Exception:
            pass


def sort_batch_id(batch_id: Union[int, str]):
    """Priority of batch_id"""
    # TODO: avoid sort the batch_id every time
    if isinstance(batch_id, int):
        return (batch_id, 0)
    else:
        match = re.match(r"^(\d+)", batch_id)
        if match:
            num = int(match.group(1))
            return (num, 1)
        else:
            return (float("inf"), 1)


class Scheduler:
    """Scheduler for rollout tasks."""

    def __init__(
        self,
        config: Config,
        rollout_model: List[InferenceModel],
        auxiliary_models: Optional[List[List[InferenceModel]]] = None,
    ):
        self.logger = get_logger(__name__)
        self.config = config
        self.rollout_model = rollout_model
        self.auxiliary_models = auxiliary_models or []
        self.namespace = ray.get_runtime_context().namespace
        self.timeout = config.explorer.max_timeout
        self.max_retry_times = config.explorer.max_retry_times
        self.running = False

        self.runner_num = len(rollout_model) * config.explorer.runner_per_model
        self.runners: Dict[int, RunnerWrapper] = dict()
        self.idle_runners = set()  # runner_id
        self.busy_runners = dict()  # runner_id -> (task, batch_id)

        self.pending_tasks_heap = []
        self.pending_tasks: Dict[Union[int, str], deque] = defaultdict(deque)  # batch_id -> tasks
        self.running_tasks: Dict[Union[int, str], set[asyncio.Future]] = defaultdict(
            set
        )  # batch_id -> futures
        self.completed_tasks: Dict[Union[int, str], deque[Status]] = defaultdict(
            deque
        )  # batch_id -> results

        self.scheduler_task: Optional[asyncio.Task] = None
        self.running = False

        self.total_scheduled = 0
        self.total_completed = 0

    def _create_runner(
        self,
        runner_id: int,
    ):
        runner = RunnerWrapper(
            runner_id=runner_id,
            rollout_model=self.rollout_model[runner_id % len(self.rollout_model)],
            auxiliary_models=[
                self.auxiliary_models[j][runner_id % len(self.auxiliary_models[j])]
                for j in range(len(self.auxiliary_models))
            ],
            config=self.config,
        )
        self.runners[runner_id] = runner
        self.idle_runners.add(runner_id)

    def _restart_runner(self, runner_id: int):
        """Restart a runner."""
        self.runners[runner_id].restart_runner()

        if runner_id in self.busy_runners:
            task, idx = self.busy_runners.pop(runner_id)
            self.logger.warning(
                f"Runner {runner_id} failed to run task at batch_id {idx}: {task.raw_task}"
            )

        self.idle_runners.add(runner_id)
        self.logger.info(f"Runner {runner_id} restarted.")

    async def _scheduler_loop(self) -> None:
        self.logger.info("Scheduler loop started.")
        while self.running:
            try:
                await self._schedule_pending_tasks()
                await self._check_completed_tasks()
                await asyncio.sleep(0.01)
            except Exception:
                self.logger.error(f"Error in scheduler loop:\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)
        self.logger.info("Scheduler loop stopped.")

    async def _schedule_pending_tasks(self) -> None:
        if not self.idle_runners:
            return

        # TODO: Support more advanced scheduling strategies
        for batch_id in sorted(self.pending_tasks.keys(), key=sort_batch_id):
            task_queue = self.pending_tasks[batch_id]

            while task_queue and self.idle_runners:
                task = task_queue.pop()
                runner_id = self.idle_runners.pop()
                self.busy_runners[runner_id] = (task, batch_id)
                self.running_tasks[batch_id].add(
                    asyncio.create_task(self.runners[runner_id].run_with_retry(task))
                )

            if not task_queue:
                del self.pending_tasks[batch_id]

    async def _check_completed_tasks(self) -> None:
        for batch_id in list(self.running_tasks.keys()):
            futures = self.running_tasks[batch_id]

            for future in list(futures):
                if future.done():
                    futures.remove(future)
                    try:
                        task_result, runner_id = await future
                        self.completed_tasks[batch_id].appendleft(task_result)
                        self.busy_runners.pop(runner_id)
                        self.idle_runners.add(runner_id)

                        self.logger.debug(
                            f"Task completed (batch_id {batch_id}), success: {task_result.ok}"
                        )

                    except Exception as e:
                        self.logger.error(f"Error getting task result: {e}")

            if not futures:
                del self.running_tasks[batch_id]

    def _clear_timeout_tasks(self, batch_id: Union[int, str]) -> None:
        if batch_id in self.pending_tasks:
            self.logger.info(f"Clear timeout pending tasks at batch_id {batch_id}.")
            del self.pending_tasks[batch_id]
        if batch_id in self.running_tasks:
            self.logger.info(f"Clear timeout running tasks at batch_id {batch_id}.")
            for future in self.running_tasks[batch_id]:
                future.cancel()
            del self.running_tasks[batch_id]

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        for i in range(self.runner_num):
            self._create_runner(i)
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        for _, runner in self.runners.items():
            await runner.runner.__ray_ready__.remote()
        self.logger.info(f"Starting Scheduler with {self.runner_num} runners")

    async def stop(self) -> None:
        if not self.running:
            return

        self.running = False
        all_running_futures = []
        for futures in self.running_tasks.values():
            all_running_futures.extend(futures)

        if all_running_futures:
            self.logger.info(f"Waiting for {len(all_running_futures)} running tasks to complete...")
            await asyncio.gather(*all_running_futures, return_exceptions=True)

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Scheduler stopped")

    def schedule(self, tasks: List[Task], batch_id: Union[int, str]) -> None:
        """Schedule the provided tasks.

        Args:
            tasks (`List[Task]`): The tasks to schedule.
            batch_id (`Union[int, str]`): The id of provided tasks. It should be an integer or a string
                starting with an integer (e.g., 123, "123/my_task")
        """
        if not tasks:
            return
        for task in tasks:
            self.pending_tasks[batch_id].appendleft(task)

    async def get_results(
        self,
        batch_id: Union[int, str],
        min_num: Optional[int] = None,
        timeout: Optional[float] = None,
        clear_timeout_tasks: bool = True,
    ) -> List[Status]:
        """Get the result of tasks at the specific batch_id.

        Args:
            batch_id (`Union[int, str]`): Only wait for tasks at this batch.
            min_num (`int`): The minimum number of tasks to wait for. If `None`, wait for all tasks at `batch_id`.
            timeout (`float`): The timeout for waiting for tasks to finish. If `None`, wait for default timeout.
            clear_timeout_tasks (`bool`): Whether to clear timeout tasks.
        """
        timeout = timeout or self.timeout
        start_time = time.time()
        if min_num is None:
            min_num = 0
            if batch_id in self.pending_tasks:
                min_num += len(self.pending_tasks[batch_id])
            if batch_id in self.running_tasks:
                min_num += len(self.running_tasks[batch_id])
            if batch_id in self.completed_tasks:
                min_num += len(self.completed_tasks[batch_id])

        self.logger.debug(f"Waiting for {min_num} tasks to complete...")

        while time.time() - start_time < timeout:
            completed_count = len(self.completed_tasks[batch_id])
            if completed_count >= min_num:
                break
            await asyncio.sleep(0.1)

        if time.time() - start_time > timeout:
            self.logger.error(f"Timed out waiting for tasks to complete after {timeout} seconds")
            if clear_timeout_tasks:
                self._clear_timeout_tasks(batch_id=batch_id)
                for runner_id in list(self.busy_runners.keys()):
                    if self.busy_runners[runner_id][1] == batch_id:
                        self._restart_runner(runner_id)

        results = []
        for _ in range(min_num):
            if len(self.completed_tasks[batch_id]) > 0:
                results.append(self.completed_tasks[batch_id].pop())

        if not self.completed_tasks[batch_id]:
            del self.completed_tasks[batch_id]

        completed_count = len(results)
        if completed_count < min_num:
            self.logger.warning(
                f"Timeout reached, only {completed_count}/{min_num} tasks completed"
            )

        return results

    def has_step(self, batch_id: Union[int, str]) -> bool:
        return (
            batch_id in self.completed_tasks
            or batch_id in self.pending_tasks
            or batch_id in self.running_tasks
        )

    async def wait_all(
        self, timeout: Optional[float] = None, clear_timeout_tasks: bool = True
    ) -> None:
        """Wait for all tasks to complete without poping results. If timeout reached, raise TimeoutError.

        Args:
            timeout (`float`): timeout in seconds.
            clear_timeout_tasks (`bool`): Whether to clear timeout tasks.
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        self.logger.debug("Waiting for all tasks to complete...")

        while time.time() - start_time < timeout:
            has_pending = bool(self.pending_tasks)
            has_running = bool(self.running_tasks)

            if not has_pending and not has_running:
                self.logger.debug("All tasks completed successfully")
                return

            pending_count = sum(len(tasks) for tasks in self.pending_tasks.values())
            running_count = sum(len(futures) for futures in self.running_tasks.values())

            self.logger.debug(f"Pending tasks: {pending_count}, Running tasks: {running_count}")
            await asyncio.sleep(0.1)

        pending_count = sum(len(tasks) for tasks in self.pending_tasks.values())
        running_count = sum(len(futures) for futures in self.running_tasks.values())
        error_msg = f"Timeout after {timeout} seconds. Still have {pending_count} pending tasks and {running_count} running tasks."
        self.logger.error(error_msg)

        if clear_timeout_tasks:
            for batch_id in self.pending_tasks.keys() | self.running_tasks.keys():
                self._clear_timeout_tasks(batch_id)
            busy_runner_ids = list(self.busy_runners.keys())
            for runner_id in busy_runner_ids:
                self._restart_runner(runner_id)

        raise TimeoutError(error_msg)
