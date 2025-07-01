# -*- coding: utf-8 -*-
"""The explorer module"""
from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from typing import List, Optional

import torch

from trinity.algorithm.algorithm_manager import AlgorithmManager
from trinity.buffer import get_buffer_writer
from trinity.buffer.buffer import get_buffer_reader
from trinity.common.config import Config
from trinity.common.constants import (
    ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
    RunningStatus,
    SyncMethod,
)
from trinity.common.models import create_inference_models
from trinity.common.models.utils import (
    get_checkpoint_dir_with_step_num,
    load_state_dict,
)
from trinity.explorer.runner_pool import RunnerPool
from trinity.manager.manager import CacheManager
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR


class Explorer:
    """Responsible for exploring the taskset."""

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.cache = CacheManager(config)
        explorer_meta = self.cache.load_explorer()
        self.explore_step_num = explorer_meta.get("latest_iteration", 0)
        self.config = config
        self.algorithm_manager = AlgorithmManager(config)
        self.models, self.auxiliary_models = create_inference_models(config)
        if self.config.mode != "bench":
            self.experience_buffer = get_buffer_writer(
                self.config.buffer.explorer_output,  # type: ignore
                self.config.buffer,
            )
            self.experience_buffer.acquire()
        self.config.buffer.explorer_input.taskset.index = explorer_meta.get("latest_task_index", 0)
        self.taskset = get_buffer_reader(
            self.config.buffer.explorer_input.taskset, self.config.buffer
        )
        self.runner_pool = self._init_runner_pool()
        self.monitor = MONITOR.get(self.config.monitor.monitor_type)(
            project=self.config.project,
            name=self.config.name,
            role=self.config.explorer.name,
            config=config,
        )
        self.batch_size = config.buffer.batch_size
        self.update_interval = (
            self.config.synchronizer.sync_interval * self.config.buffer.batch_size
        )
        self.use_checkpoint_weights_update = (
            self.config.synchronizer.sync_method == SyncMethod.CHECKPOINT
        )
        self.eval_explore_step_num = None

        # For checkpoint weights update
        # Use explorer to periodically load the latest model weights and
        # boradcast to all rollout models
        if self.use_checkpoint_weights_update:
            self.old_checkpoint = None
            self.state_dict = {}
        else:  # nccl mode
            self.state_dict_meta = []
        self.status = RunningStatus.RUNNING
        self.logger.info("Finished initializing Explorer.")

    async def setup_weight_sync_group(
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
                explorer_name=self.config.explorer.name,
                timeout=self.config.synchronizer.sync_timeout,
                update_with_checkpoint=self.use_checkpoint_weights_update,
                state_dict_meta=state_dict_meta,
            )
            for i, model in enumerate(self.models)
        ]
        await asyncio.gather(*refs)

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

    async def _update_model_weight(self, step_num: int, state_dict: dict) -> None:
        # TODO: update model weight
        self.state_dict = state_dict
        if self.state_dict_meta is None:
            update_weight_args_list = []
            for name, param in state_dict.items():
                update_weight_args_list.append((name, str(param.dtype), tuple(param.shape)))
            self.state_dict_meta = update_weight_args_list
        else:
            update_weight_args_list = None
        await asyncio.gather(
            *[model.sync_model.remote(step_num, update_weight_args_list) for model in self.models]
        )
        self.state_dict.clear()

    async def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> None:
        # TODO: support more checkpoint types
        try:
            checkpoint_dir, checkpoint_step_num = get_checkpoint_dir_with_step_num(
                checkpoint_root_path=self.config.checkpoint_job_dir,
                trainer_type=self.config.trainer.trainer_type,
                step_num=step_num,
            )
            if checkpoint_dir == self.old_checkpoint:
                return
            model_weights = load_state_dict(os.path.join(checkpoint_dir, "actor"))
            await self._update_model_weight(checkpoint_step_num, model_weights)
            self.old_checkpoint = checkpoint_dir
        except Exception as e:
            self.logger.warning(f"Fail to load checkpoint: {e}")

    async def _nccl_weights_update(self):
        assert self.state_dict_meta is not None
        await asyncio.gather(
            *[model.sync_model.remote(self.explore_step_num) for model in self.models]
        )

    async def prepare(self) -> None:
        """Preparation before running."""
        if self.use_checkpoint_weights_update:
            master_address, master_port = await self.models[0].get_available_address.remote()
            await self.setup_weight_sync_group(master_address, master_port)

    async def get_weight(self, name: str) -> torch.Tensor:
        """Get the weight of the loaded model (For checkpoint weights update)."""
        return self.state_dict[name]

    async def explore(self) -> str:
        """
        The dreamming loop for explorer and trainer.
                 | <----------------------------------------- one period ----------------------------------------------> |
        explorer | <-- step_1 --> | <-- step_2 --> | ... | <-- step_n --> | <-- eval --> | <-- [idle] --> | <-- sync --> |
         trainer | <-- idle --> | <-- step_1 --> | <-- step_2 --> | ... | <-- step_n --> | <-- [idle] --> | <-- sync --> |
        """
        self.eval_explore_step_num = None
        while True:
            try:
                self.logger.info(f"Explore step {self.explore_step_num + 1} started.")
                if (
                    self.eval_explore_step_num is None
                    and self.explore_step_num % self.config.explorer.eval_interval == 0
                ):
                    self.eval_explore_step_num = self.explore_step_num
                explore_contionue = self.explore_step()
                if not explore_contionue:
                    break
                if self.need_sync():
                    self.wait_for_workflow_done()
                    await self.sync_weight()
            except Exception as e:
                self.logger.error(f"Error in Explorer: {e}")
                break
        self.logger.info("--------------------\n> Explorer finished.\n--------------------")
        return self.config.explorer.name

    def explore_step(self) -> bool:
        algo_config = self.algorithm_manager.get_current_algorithm_config(self.explore_step_num + 1)
        # skip warmup
        if algo_config.algorithm_type == "sft":
            self.explore_step_num += 1
            return True
        try:
            tasks = self.taskset.read()
        except StopIteration:
            self.logger.warning("No more tasks to explore. Stop exploring.")
            self.cache.save_explorer(
                current_step=self.explore_step_num,
                current_task_index=self.explore_step_num * self.config.buffer.batch_size,
            )
            self.status = RunningStatus.STOPPED
            self.wait_for_workflow_done()
            self.experience_buffer.release()
            return False
        self.runner_pool.run_tasks(tasks)
        self.explore_step_num += 1
        return True

    def need_sync(self) -> bool:
        if self.explore_step_num <= self.config.synchronizer.sync_offset:
            return False
        return (
            self.explore_step_num - self.config.synchronizer.sync_offset
        ) % self.config.synchronizer.sync_interval == 0

    def eval(self, eval_explore_step_num: int):
        """Evaluation on all evaluation data samples."""
        if len(self.config.buffer.explorer_input.eval_tasksets) == 0:
            self.logger.warning("No evaluation data samples. Skip evaluation.")
            return
        self.logger.info(f"Evaluation at step {eval_explore_step_num} started.")
        all_st = time.time()
        log_metrics = {}
        for eval_taskset_config in self.config.buffer.explorer_input.eval_tasksets:
            self.logger.info(
                f"Evaluation on {eval_taskset_config.name} at step {eval_explore_step_num} started."
            )
            eval_taskset = get_buffer_reader(eval_taskset_config, self.config.buffer)
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
        self.monitor.log(log_metrics, step=eval_explore_step_num)  # type: ignore
        self.logger.info(f"Evaluation at step {eval_explore_step_num} finished.")

    async def benchmark(self) -> bool:
        """Benchmark the model checkpoints."""
        # benchmark on the latest checkpoint
        if self.config.explorer.eval_on_latest_checkpoint:
            await self._checkpoint_weights_update()
            self.eval(self.explore_step_num)
            return True

        # benchmark on base model
        self.eval(0)
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
            await self._checkpoint_weights_update(step_num=step_num)
            self.eval(step_num)
        return True

    def wait_for_workflow_done(self) -> None:
        """Wait for workflow to finish."""
        all_metrics = defaultdict(list)
        # wait for all tasks of this step to finish
        while self.runner_pool.has_next():
            status_list = self.runner_pool.get_next_unorder()
            if not isinstance(status_list, list):
                status_list = [status_list]
            for status in status_list:
                if not status.ok:
                    self.logger.error(f"Error when running task: {status.message}")
                    # submit another task to replace the failed task
                    try:
                        tasks = self.taskset.read(batch_size=1)
                    except StopIteration:
                        self.logger.warning("No more tasks in taskset. Stop retrying.")
                        return
                    self.runner_pool.run_tasks(tasks)
                else:
                    for metric_name, metric_value in status.metric.items():
                        all_metrics[metric_name].append(metric_value)
        # eval
        if self.eval_explore_step_num is not None:
            self.eval(self.eval_explore_step_num)
            self.eval_explore_step_num = None
        # calculate metrics
        log_metrics = self.monitor.calculate_metrics(all_metrics, prefix="rollout")  # type: ignore
        self.monitor.log(log_metrics, step=self.explore_step_num)
        self.logger.info(f"Explore step {self.explore_step_num} finished.")

    async def sync_weight(self) -> None:
        """Synchronize model weights."""
        # call this method before training start to load the latest model weights
        self.logger.info(f"Explorer sync weights at step {self.explore_step_num}.")
        self.status = RunningStatus.WAITING_SYNC
        if self.use_checkpoint_weights_update:
            await self._checkpoint_weights_update()
        else:  # nccl weights update
            await self._nccl_weights_update()
        # save explore checkpoint
        self.cache.save_explorer(
            current_step=self.explore_step_num,
            current_task_index=self.explore_step_num * self.config.buffer.batch_size,
        )
        self.status = RunningStatus.RUNNING
        self.logger.info(f"Explorer sync at step {self.explore_step_num} finished")

    async def running_status(self) -> RunningStatus:
        return self.status

    def flush_log(self, step: int) -> None:
        """Flush the log of the current step."""
        self.monitor.log({}, step=step, commit=True)

    def shutdown(self) -> None:
        self.monitor.close()
