# -*- coding: utf-8 -*-
"""The explorer module"""
import os
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import ray
import torch

from trinity.buffer import get_buffer_writer
from trinity.common.config import Config
from trinity.common.constants import (
    ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
    SyncMethod,
    TaskType,
)
from trinity.common.models import create_rollout_models
from trinity.common.models.utils import (
    get_checkpoint_dir_with_step_num,
    load_state_dict,
)
from trinity.common.task import TaskSet
from trinity.explorer.runner_pool import RunnerPool
from trinity.manager.manager import CacheManager
from trinity.utils.log import get_logger
from trinity.utils.monitor import Monitor


@ray.remote(name="explorer", concurrency_groups={"get_weight": 32, "setup_weight_sync_group": 1})
class Explorer:
    """Responsible for exploring the taskset."""

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.cache = CacheManager(config)
        explorer_meta = self.cache.load_explorer()
        self.step_num = explorer_meta.get("latest_iteration", 0)
        self.config = config
        self.models = create_rollout_models(config)
        self.experience_buffer = get_buffer_writer(
            self.config.buffer.train_dataset,  # type: ignore
            self.config.buffer,
        )
        self.taskset = TaskSet.load(
            self.config.data, explorer_meta.get("latest_task_index", 0), TaskType.EXPLORE
        )
        if self.config.data.eval_split:
            self.eval_taskset = TaskSet.load(self.config.data, task_type=TaskType.EVAL)
        else:
            self.eval_taskset = None
        self.task_iter = None
        self.runner_pool = self._init_runner_pool()
        self.monitor = Monitor(
            project=self.config.monitor.project,
            name=self.config.monitor.name,
            role="explorer",
            config=config,
        )
        self.max_pending_task_num = self.config.explorer.runner_num
        self.max_waiting_steps = max(1, int(self.config.explorer.max_waiting_steps))
        self.batch_size = config.data.batch_size
        self.update_interval = self.config.synchronizer.sync_interval * self.config.data.batch_size
        self.use_checkpoint_weights_update = (
            self.config.synchronizer.sync_method == SyncMethod.CHECKPOINT
        )

        # For checkpoint weights update
        # Use explorer to periodically load the latest model weights and
        # boradcast to all rollout models
        if self.use_checkpoint_weights_update:
            self.old_checkpoint = None
            self.state_dict = {}
        else:  # nccl mode
            self.state_dict_meta = []
        self.logger.info("Finished initializing Explorer.")

    @ray.method(concurrency_group="setup_weight_sync_group")
    def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        # In checkpoint mode, we use explorer to store the model weights which has no rank
        base_offset = 0 if self.use_checkpoint_weights_update else 1
        world_size = len(self.models) * self.config.explorer.tensor_parallel_size + base_offset
        self.logger.info(
            f"Initialize process group for weight synchronization, "
            f"master_address={master_address}, master_port={master_port}, "
            f"world_size={world_size}, rank_offset={base_offset}"
        )
        self.state_dict_meta = state_dict_meta
        refs = [
            model.init_process_group.remote(
                master_address=master_address,
                master_port=master_port,
                rank_offset=i * self.config.explorer.tensor_parallel_size + base_offset,
                world_size=world_size,
                group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                backend=self.config.explorer.backend,
                timeout=self.config.synchronizer.sync_timeout,
                update_with_checkpoint=self.use_checkpoint_weights_update,
            )
            for i, model in enumerate(self.models)
        ]
        ray.get(refs)

    def _init_runner_pool(self) -> RunnerPool:
        if self.config.explorer.engine_type != "vllm_async":
            # sync model requires the same number of runners as the number of models
            self.config.explorer.runner_num = self.config.explorer.engine_num
            self.logger.info(
                "Sync vLLM model requires the same number of runners as the number of models"
            )
        if self.config.explorer.runner_num < self.config.explorer.engine_num:
            self.config.explorer.runner_num = self.config.explorer.engine_num
            self.logger.info(
                f"Number of Runners is less than number of models, set to {self.config.explorer.runner_num}"
            )
        self.logger.info(f"Setup {self.config.explorer.runner_num} WorkflowRunners")
        return RunnerPool(self.config, self.models)

    def _update_model_weight(self, state_dict: dict) -> None:
        # TODO: update model weight
        self.state_dict = state_dict
        update_weight_args_list = []
        for name, param in state_dict.items():
            update_weight_args_list.append((name, param.dtype, param.shape))
        ray.get([model.sync_model.remote(update_weight_args_list) for model in self.models])
        self.state_dict.clear()

    def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> None:
        # TODO: support more checkpoint types
        try:
            checkpoint_dir = get_checkpoint_dir_with_step_num(
                checkpoint_root_path=self.config.model.checkpoint_path,
                trainer_type=self.config.trainer.trainer_type,
                step_num=step_num,
            )
            if checkpoint_dir == self.old_checkpoint:
                return
            model_weights = load_state_dict(os.path.join(checkpoint_dir, "actor"))
            self._update_model_weight(model_weights)
            self.old_checkpoint = checkpoint_dir
        except Exception as e:
            self.logger.error(f"Error when loading state_dict: {e}")

    def _nccl_weights_update(self):
        ray.get([model.sync_model.remote(self.state_dict_meta) for model in self.models])

    def prepare(self) -> None:
        """Preparation before running."""
        if self.use_checkpoint_weights_update:
            master_address, master_port = ray.get(self.models[0].get_address.remote())
            self.setup_weight_sync_group(master_address, master_port)

    @ray.method(concurrency_group="get_weight")
    def get_weight(self, name: str) -> torch.Tensor:
        """Get the weight of the loaded model (For checkpoint weights update)."""
        return self.state_dict[name]

    def explore(self) -> None:
        """Explore the entire dataset."""
        while True:
            explore_status, explore_iter = self.explore_one_period()
            if not explore_status:
                break
            self.sync_weight()
            if explore_iter % self.config.explorer.eval_interval == 0:
                self.eval()
                self.logger.info("Evaluation finished.")
        self.logger.info("Explorer finished.")

    def explore_one_period(self) -> Tuple[bool, int]:
        """Explore for one period.

        Different from `explore()` which consumes all tasks in the task set,
        `explore_one_period()` only consume `sync_interval * batch_size`
        number of tasks.
        Returns:
            explore_status: whether there are more tasks to explore.
            explore_step_num: the number of explore steps
        """
        if self.task_iter is None:
            self.task_iter = iter(self.taskset)
        task_num_per_period = self.config.synchronizer.sync_interval * self.config.data.batch_size

        st = time.time()
        all_metrics = defaultdict(list)

        # submit tasks of this step
        try:
            tasks = [next(self.task_iter) for _ in range(task_num_per_period)]  # type: ignore
            self.runner_pool.run_tasks(tasks)
        except StopIteration:
            self.experience_buffer.finish()
            self.logger.warning("No more tasks in the task set. Stop exploring.")
            return False, self.step_num

        # wait for all tasks of this step to finish
        while self.runner_pool.has_next():
            status_list = self.runner_pool.get_next_unorder()
            if not isinstance(status_list, list):
                status_list = [status_list]
            for status in status_list:
                if not status.ok:
                    self.logger.error(f"Error when running task: {status.message}")
                    try:
                        # submit another task to replace the failed task
                        self.runner_pool.run_tasks(next(self.task_iter))  # type: ignore
                    except StopIteration:
                        self.logger.warning("No more tasks in the task set. Stop exploring.")
                        return False, self.step_num
                else:
                    for metric_name, metric_value in status.metric.items():
                        all_metrics[metric_name].append(metric_value)

        # calculate metrics
        log_metrics = self.monitor.calculate_metrics(all_metrics, prefix="rollout")  # type: ignore
        log_metrics["rollout/step_time"] = time.time() - st
        self.step_num += self.config.synchronizer.sync_interval
        self.monitor.log(log_metrics, step=self.step_num)

        # save explore checkpoint
        self.cache.save_explorer(
            current_step=self.step_num,
            current_task_index=self.taskset.index,
        )

        self.logger.info(f"Explore step {self.step_num} finished.")
        return True, self.step_num

    def eval(self) -> Tuple[bool, int]:
        """Evaluation on all evaluation data samples."""
        if self.eval_taskset is None:
            self.logger.warning("No evaluation data samples. Skip evaluation.")
            return True, self.step_num
        self.logger.info("Evaluation started.")
        st = time.time()
        all_metrics = defaultdict(list)

        tasks = [task for task in self.eval_taskset]
        self.runner_pool.run_tasks(tasks)

        while self.runner_pool.has_next():
            # TODO: use unordered queue to avoid blocking
            status_list = self.runner_pool.get_next_unorder()
            if not isinstance(status_list, list):
                status_list = [status_list]
            for status in status_list:
                if not status.ok:
                    self.logger.error(f"Error when running task: {status.message}")
                else:
                    for metric_name, metric_value in status.metric.items():
                        all_metrics[metric_name].append(metric_value)

        log_metrics = self.monitor.calculate_metrics(all_metrics, prefix="eval")  # type: ignore
        log_metrics["eval/total_time"] = time.time() - st
        self.monitor.log(log_metrics, step=self.step_num)  # type: ignore
        return True, self.step_num

    def sync_weight(self) -> None:
        """Synchronize model weights."""
        # call this method before training start to load the latest model weights
        if self.use_checkpoint_weights_update:
            self._checkpoint_weights_update()
        else:  # nccl weights update
            self._nccl_weights_update()

    def flush_log(self, step: int) -> None:
        """Flush the log of the current step."""
        self.monitor.log({}, step=step, commit=True)
