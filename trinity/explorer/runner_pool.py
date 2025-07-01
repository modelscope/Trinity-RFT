"""Runner pool for running tasks in parallel. Modified from ray.util.actor_pool.ActorPool."""
import random
from typing import List, Optional, Tuple, Union

import ray

from trinity.common.config import Config
from trinity.common.models.model import InferenceModel
from trinity.common.workflows import Task
from trinity.explorer.workflow_runner import Status, WorkflowRunner
from trinity.utils.log import get_logger


class RunnerPool:
    """A pool of WorkflowRunner.

    The RunnerPool will automatically handle the exceptions during the workflow
    and retry when the workflow fails or timeout. The number of max retries is
    set in `config.explorer.max_retry_times` and the max timeout is set in
    `config.explorer.max_timeout`.
    """

    def __init__(
        self,
        config: Config,
        models: List[InferenceModel],
        auxiliary_models: Optional[List[List[InferenceModel]]] = None,
    ):
        # actors to be used
        self.logger = get_logger(__name__)
        self.config = config
        self.models = models
        self.auxiliary_models = auxiliary_models or []
        self.timeout = config.explorer.max_timeout
        self.max_retry_times = config.explorer.max_retry_times

        # get actor from future
        self._future_to_actor = {}

        # get future from index
        self._index_to_future = {}

        # next task to do
        self._next_task_index = 0

        # next task to return
        self._next_return_index = 0

        # next work depending when actors free
        self._pending_submits = []

        # create new actors
        self.engine_status = [0] * config.explorer.rollout_model.engine_num
        self.auxiliary_engine_status_list = [
            [0] * cfg.engine_num for cfg in config.explorer.auxiliary_models
        ]
        self._idle_actors = list()
        self.actor_to_engine_index = {}
        self._namespace = ray.get_runtime_context().namespace
        self._create_actors(config.explorer.runner_num)

    def _create_actors(self, num: int = 1):
        new_actors = []
        for _ in range(num):
            engine_index = self.engine_status.index(min(self.engine_status))
            selected_auxiliary_models = [
                models[engine_status.index(min(engine_status))]
                for models, engine_status in zip(
                    self.auxiliary_models, self.auxiliary_engine_status_list
                )
            ]
            new_actor = (
                ray.remote(WorkflowRunner)
                .options(
                    namespace=self._namespace,
                    scheduling_strategy="SPREAD",
                    runtime_env={"env_vars": self.config.explorer.env_vars},
                )
                .remote(
                    self.config,
                    self.models[engine_index],
                    selected_auxiliary_models,
                )
            )
            new_actors.append(new_actor)
            self.engine_status[engine_index] += 1
            self.actor_to_engine_index[new_actor] = engine_index
        for actor in new_actors:
            self._return_actor(actor)

    def _kill_actors(self, actors):
        if not isinstance(actors, list):
            actors = [actors]

        for actor in actors:
            release_engine_index = self.actor_to_engine_index[actor]
            self.engine_status[release_engine_index] -= 1
            del self.actor_to_engine_index[actor]
            ray.kill(actor)

    def _run_task(self, task: Task, retry_times: int = 0) -> None:
        """Run a task in the pool.

        Arguments:
            task: A task to run.
            retry_times: The current retry times of the task.
        """
        if self._idle_actors:
            actor = self._idle_actors.pop()
            future = actor.run_task.remote(task)
            future_key = tuple(future) if isinstance(future, list) else future
            self._future_to_actor[future_key] = (task, actor, retry_times)
            self._index_to_future[self._next_task_index] = future
            self._next_task_index += 1
        else:
            self._pending_submits.append((task, retry_times))

    def run_tasks(self, tasks: Union[List[Task], Task]) -> None:
        """Schedule a list of tasks to run in the pool.

        Arguments:
            tasks: A list of tasks.
        """
        if isinstance(tasks, Task):
            tasks = [tasks]
        for task in tasks:
            self._run_task(task, 0)

    def has_next(self):
        """Returns whether there are any pending results to return.

        Returns:
            True if there are any pending results not yet returned.
        """
        return bool(self._future_to_actor)

    def _handle_single_future(self, future, is_timeout) -> Tuple[Status, Task, int]:
        future_key = tuple(future) if isinstance(future, list) else future
        t, a, r = self._future_to_actor.pop(future_key)

        if is_timeout:
            # when timeout, restart the actor
            self.logger.warning(f"Workflow {t.task_desc} Timeout.")

            # kill the actor and update engine status
            self._kill_actors(a)

            # start a new actor
            self._create_actors(num=1)

            return_status = Status(
                False, metric={"time_per_task": self.timeout}, message="Workflow Timeout."
            )
        else:
            self._return_actor(a)
            try:
                return_status = ray.get(future)
            except Exception as e:
                self.logger.error(f"Error when running task: {e}")
                return_status = Status(
                    False,
                    metric={"time_per_task": self.timeout},
                    message=f"Error when running task: {e}",
                )
        return return_status, t, r

    def get_next_unorder(self) -> List[Status]:
        """Returns the next pending result unorder.

        Returns:
            The return status of the next task.
        """
        if not self.has_next():
            raise StopIteration("No more results to get")
        is_timeout = False
        res, _ = ray.wait(list(self._future_to_actor), num_returns=1, timeout=self.timeout)
        if not res:
            is_timeout = True
            future_list = list(self._future_to_actor)
        else:
            future_list = res

        return_status_list = list()
        for future in future_list:
            return_status, t, r = self._handle_single_future(future, is_timeout)

            if not return_status.ok:
                if r >= self.max_retry_times:
                    return_status_list.append(
                        Status(
                            False,
                            metric={"retry_times": r + 1},
                            message=f"{return_status.message}\nWorkflow Retry Times Exceeded.",
                        )
                    )
                else:
                    self.logger.info(f"Retry Workflow {t.task_desc}.")
                    self._run_task(t, r + 1)
            else:
                return_status_list.append(return_status)

        return return_status_list if return_status_list else self.get_next_unorder()

    # todo: this function may be discarded in the next version
    def get_next(self) -> Status:
        """Returns the next pending result in order.

        This returns the next task result, blocking for up to
        the specified timeout until it is available.

        Returns:
            The return status of the next task.
        """
        if not self.has_next():
            raise StopIteration("No more results to get")
        future = self._index_to_future[self._next_return_index]
        is_timeout = False
        res, _ = ray.wait([future], timeout=self.timeout)
        if not res:
            is_timeout = True
        del self._index_to_future[self._next_return_index]
        self._next_return_index += 1

        future_key = tuple(future) if isinstance(future, list) else future
        t, a, r = self._future_to_actor.pop(future_key)

        if is_timeout:
            # when timeout, restart the actor
            self.logger.warning(f"Workflow {t.task_desc} Timeout.")
            ray.kill(a)
            # TODO: balance the model
            self._return_actor(
                ray.remote(WorkflowRunner)
                .options(
                    namespace=self._namespace,
                    scheduling_strategy="SPREAD",
                )
                .remote(
                    self.config,
                    self.models[
                        random.randint(0, self.config.explorer.rollout_model.engine_num - 1)
                    ],
                )
            )
            return_status = Status(
                False, metric={"time_per_task": self.timeout}, message="Workflow Timeout."
            )
        else:
            self._return_actor(a)
            try:
                return_status = ray.get(future)
            except Exception as e:
                self.logger.error(f"Error when running task: {e}")
                return_status = Status(
                    False,
                    metric={"time_per_task": self.timeout},
                    message=f"Error when running task: {e}",
                )

        if not return_status.ok:
            if r >= self.max_retry_times:
                return Status(
                    False,
                    metric={"retry_times": r + 1},
                    message=f"{return_status.message}\nWorkflow Retry Times Exceeded.",
                )
            else:
                self.logger.info(f"Retry Workflow {t.task_desc}.")
                self._run_task(t, r + 1)
            return self.get_next()
        else:
            return return_status

    def _return_actor(self, actor):
        try:
            ray.get(actor.is_alive.remote())
            self._idle_actors.append(actor)
        except Exception:
            self.logger.info("The actor is not alive, restart a new actor")
            self._kill_actors(actor)
            self._create_actors(num=1)

        if self._pending_submits:
            self._run_task(*self._pending_submits.pop(0))

    def has_free(self):
        """Returns whether there are any idle actors available.

        Returns:
            True if there are any idle actors and no pending submits.
        """
        return len(self._idle_actors) > 0 and len(self._pending_submits) == 0

    def pop_idle(self):
        """Removes an idle actor from the pool.

        Returns:
            An idle actor if one is available.
            None if no actor was free to be removed.
        """
        if self.has_free():
            return self._idle_actors.pop()
        return None
