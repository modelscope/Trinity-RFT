# -*- coding: utf-8 -*-
"""The explorer module"""
import os
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import ray
import torch

from trinity.algorithm.algorithm_manager import AlgorithmManager
from trinity.buffer import get_buffer_writer
from trinity.buffer.buffer import get_buffer_reader
from trinity.common.config import Config
from trinity.common.constants import ROLLOUT_WEIGHT_SYNC_GROUP_NAME, SyncMethod
from trinity.common.models import create_inference_models
from trinity.common.models.utils import (
    get_checkpoint_dir_with_step_num,
    load_state_dict,
)
from trinity.explorer.runner_pool import RunnerPool
from trinity.manager.manager import CacheManager
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR


@ray.remote(name="explorer", concurrency_groups={"get_weight": 32, "setup_weight_sync_group": 1})
class Explorer:
    """Responsible for exploring the taskset."""

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.cache = CacheManager(config)
        explorer_meta = self.cache.load_explorer()
        self.step_num = explorer_meta.get("latest_iteration", 0)
        self.config = config
        self.algorithm_manager = AlgorithmManager(config)
        self.models, self.auxiliary_models = create_inference_models(config)
        if self.config.mode != "bench":
            self.experience_buffer = get_buffer_writer(
                self.config.buffer.explorer_output,  # type: ignore
                self.config.buffer,
            )
        self.config.buffer.explorer_input.taskset.index = explorer_meta.get("latest_task_index", 0)
        self.taskset = get_buffer_reader(
            self.config.buffer.explorer_input.taskset, self.config.buffer
        )
        self.runner_pool = self._init_runner_pool()
        self.monitor = MONITOR.get(self.config.monitor.monitor_type)(
            project=self.config.project,
            name=self.config.name,
            role="explorer",
            config=config,
        )
        self.batch_size = config.buffer.batch_size
        self.update_interval = (
            self.config.synchronizer.sync_interval * self.config.buffer.batch_size
        )
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
        world_size = (
            len(self.models) * self.config.explorer.rollout_model.tensor_parallel_size + base_offset
        )
        self.logger.info(
            f"Initialize process group for weight synchronization, "
            f"master_address={master_address}, master_port={master_port}, "
            f"world_size={world_size}, rank_offset={base_offset}"
        )
        self.state_dict_meta = state_dict_meta
        # TODO: save state_dict in models
        refs = [
            model.init_process_group.remote(
                master_address=master_address,
                master_port=master_port,
                rank_offset=i * self.config.explorer.rollout_model.tensor_parallel_size
                + base_offset,
                world_size=world_size,
                group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                timeout=self.config.synchronizer.sync_timeout,
                update_with_checkpoint=self.use_checkpoint_weights_update,
                state_dict_meta=state_dict_meta,
            )
            for i, model in enumerate(self.models)
        ]
        ray.get(refs)

    def _init_runner_pool(self) -> RunnerPool:
        if self.config.explorer.rollout_model.engine_type != "vllm_async":
            # sync model requires the same number of runners as the number of models
            self.config.explorer.runner_num = self.config.explorer.rollout_model.engine_num
            self.logger.info(
                "Sync vLLM model requires the same number of runners as the number of models"
            )
        if self.config.explorer.runner_num < self.config.explorer.rollout_model.engine_num:
            self.config.explorer.runner_num = self.config.explorer.rollout_model.engine_num
            self.logger.info(
                f"Number of Runners is less than number of models, set to {self.config.explorer.runner_num}"
            )
        self.logger.info(f"Setup {self.config.explorer.runner_num} WorkflowRunners")
        return RunnerPool(self.config, self.models, self.auxiliary_models)

    def _update_model_weight(self, state_dict: dict) -> None:
        # TODO: update model weight
        self.state_dict = state_dict
        if self.state_dict_meta is None:
            update_weight_args_list = []
            for name, param in state_dict.items():
                update_weight_args_list.append((name, str(param.dtype), tuple(param.shape)))
            self.state_dict_meta = update_weight_args_list
        else:
            update_weight_args_list = None
        ray.get([model.sync_model.remote(update_weight_args_list) for model in self.models])
        self.state_dict.clear()

    def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> None:
        # TODO: support more checkpoint types
        try:
            checkpoint_dir = get_checkpoint_dir_with_step_num(
                checkpoint_root_path=self.config.checkpoint_job_dir,
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
        assert self.state_dict_meta is not None
        ray.get([model.sync_model.remote() for model in self.models])

    def prepare(self) -> None:
        """Preparation before running."""
        if self.use_checkpoint_weights_update:
            master_address, master_port = ray.get(self.models[0].get_available_address.remote())
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
        # skip for sft
        algo_config = self.algorithm_manager.get_current_algorithm_config(self.step_num + 1)
        if algo_config.algorithm_type == "sft":
            for _ in range(self.config.synchronizer.sync_interval):
                self.step_num += 1
                if self.algorithm_manager.need_save(self.step_num):
                    break
            return True, self.step_num

        st = time.time()
        all_metrics = defaultdict(list)

        # submit tasks of this step
        try:
            tasks = []
            for _ in range(self.config.synchronizer.sync_interval):
                tasks.extend(self.taskset.read())
            self.runner_pool.run_tasks(tasks)  # type: ignore
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
                        self.runner_pool.run_tasks(self.taskset.read())
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
            current_task_index=self.step_num * self.config.buffer.batch_size,
            # TODO: remove current_task_index
        )

        self.logger.info(f"Explore step {self.step_num} finished.")
        return True, self.step_num

    def eval(self) -> Tuple[bool, int]:
        """Evaluation on all evaluation data samples."""
        eval_tasksets = []
        for eval_taskset_config in self.config.buffer.explorer_input.eval_tasksets:
            eval_tasksets.append(get_buffer_reader(eval_taskset_config, self.config.buffer))
        if len(eval_tasksets) == 0:
            self.logger.warning("No evaluation data samples. Skip evaluation.")
            return True, self.step_num
        self.logger.info("Evaluation started.")
        all_st = time.time()
        log_metrics = {}
        for eval_taskset in eval_tasksets:
            st = time.time()
            all_metrics = defaultdict(list)

            def wait():
                status_list = self.runner_pool.get_next_unorder()
                if not isinstance(status_list, list):
                    status_list = [status_list]
                for status in status_list:
                    if not status.ok:
                        self.logger.error(f"Error when running task: {status.message}")
                    else:
                        for metric_name, metric_value in status.metric.items():
                            all_metrics[metric_name].append(metric_value)

            while True:
                if not self.runner_pool.has_free():
                    wait()
                try:
                    self.runner_pool.run_tasks(eval_taskset.read())
                except StopIteration:
                    break
            while self.runner_pool.has_next():
                wait()
            metrics = self.monitor.calculate_metrics(all_metrics, prefix=f"eval/{eval_taskset.name}")  # type: ignore
            log_metrics.update(metrics)
            log_metrics[f"eval/{eval_taskset.name}/time"] = time.time() - st
        log_metrics["eval/total_time"] = time.time() - all_st
        self.monitor.log(log_metrics, step=self.step_num)  # type: ignore
        return True, self.step_num

    def benchmark(self) -> bool:
        """Benchmark the model checkpoints."""
        # benchmark on the latest checkpoint
        if self.config.explorer.eval_on_latest_checkpoint:
            self._checkpoint_weights_update()
            self.eval()
            return True

        # benchmark on all checkoints
        all_ckp_steps = sorted(
            [
                int(ckp.split("global_step_")[-1])
                for ckp in os.listdir(self.config.checkpoint_job_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_job_dir, ckp))
                and ckp.startswith("global_step_")
            ]
        )
        for step_num in all_ckp_steps:
            self.step_num = step_num
            self._checkpoint_weights_update(step_num=step_num)
            self.eval()
        return True

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

    def shutdown(self) -> None:
        self.monitor.close()
