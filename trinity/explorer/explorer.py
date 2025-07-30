# -*- coding: utf-8 -*-
"""The explorer module"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
from collections import deque
from typing import List, Optional

import torch

from trinity.algorithm import ADD_STRATEGY
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
from trinity.explorer.scheduler import Scheduler
from trinity.manager.manager import CacheManager
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR, gather_metrics


class Explorer:
    """Responsible for exploring the taskset."""

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.cache = CacheManager(config)
        explorer_meta = self.cache.load_explorer()
        self.explore_step_num = explorer_meta.get("latest_iteration", 0)
        self.last_sync_step = self.explore_step_num if self.explore_step_num > 0 else -1
        self.config = config
        self.algorithm_manager = AlgorithmManager(config)
        self.models, self.auxiliary_models = create_inference_models(config)
        self.experience_buffer = None
        if self.config.mode != "bench":
            self.experience_buffer = get_buffer_writer(
                self.config.buffer.explorer_output,  # type: ignore
                self.config.buffer,
            )
        self.config.buffer.explorer_input.taskset.index = explorer_meta.get("latest_task_index", 0)
        self.taskset = get_buffer_reader(
            self.config.buffer.explorer_input.taskset, self.config.buffer
        )
        self.scheduler = self._init_scheduler()
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
        self.pending_eval_tasks = deque()

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
        self._ready_to_sync_condition = asyncio.Condition()
        self.collect_experiences = self.config.explorer.collect_experiences
        self.generated_experience_cnt = 0
        if self.collect_experiences:
            assert (
                self.experience_buffer is not None
            ), "Experience buffer is required when collect_experiences is True."
            self.add_strategy = ADD_STRATEGY.get(self.config.algorithm.add_strategy)(
                self.experience_buffer, **self.config.algorithm.add_strategy_args
            )

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

    def _init_scheduler(self) -> Scheduler:
        if self.config.explorer.rollout_model.engine_type != "vllm_async":
            # sync model requires the same number of runners as the number of models
            self.config.explorer.runner_per_model = 1
            self.logger.info(
                "Sync vLLM model requires the same number of runners as the number of models"
            )
        return Scheduler(self.config, self.models, self.auxiliary_models)

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

    async def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> int:
        # TODO: support more checkpoint types
        try:
            checkpoint_dir, checkpoint_step_num = get_checkpoint_dir_with_step_num(
                checkpoint_root_path=self.config.checkpoint_job_dir,
                trainer_type=self.config.trainer.trainer_type,
                step_num=step_num,
            )
            if checkpoint_dir == self.old_checkpoint:
                return checkpoint_step_num
            model_weights = load_state_dict(os.path.join(checkpoint_dir, "actor"))
            await self._update_model_weight(checkpoint_step_num, model_weights)
            self.old_checkpoint = checkpoint_dir
            return checkpoint_step_num
        except Exception as e:
            self.logger.warning(f"Fail to load checkpoint: {e}")
            return 0

    async def _nccl_weights_update(self):
        assert self.state_dict_meta is not None
        async with self._ready_to_sync_condition:
            try:
                await asyncio.wait_for(
                    self._ready_to_sync_condition.wait_for(
                        lambda: self.status == RunningStatus.WAITING_SYNC,
                    ),
                    timeout=self.config.synchronizer.sync_timeout,
                )
            except asyncio.TimeoutError as e:
                self.logger.error(
                    f"Trainer is not ready for model weight sync in {self.config.synchronizer.sync_timeout} seconds."
                )
                raise e
        await asyncio.gather(
            *[model.sync_model.remote(self.explore_step_num) for model in self.models]
        )
        self.status = RunningStatus.RUNNING

    async def ready_to_sync(self):
        async with self._ready_to_sync_condition:
            self.status = RunningStatus.WAITING_SYNC
            self._ready_to_sync_condition.notify_all()

    async def prepare(self) -> None:
        """Preparation before running."""
        futures = [asyncio.create_task(self.scheduler.start())]
        if self.use_checkpoint_weights_update:
            master_address, master_port = await self.models[0].get_available_address.remote()
            futures.append(
                asyncio.create_task(self.setup_weight_sync_group(master_address, master_port))
            )
        asyncio.gather(*futures, return_exceptions=True)
        if self.experience_buffer:
            await self.experience_buffer.acquire()
        if self.config.explorer.eval_on_startup and self.explore_step_num == 0:
            self.eval()

    async def get_weight(self, name: str) -> torch.Tensor:
        """Get the weight of the loaded model (For checkpoint weights update)."""
        return self.state_dict[name]

    async def explore(self) -> str:
        """
        The timeline of the exploration process:
                 | <--------------------------------- one period -------------------------------------> |
        explorer | <---------------- step_1 --------------> |                                           |
                 |   | <---------------- step_2 --------------> |                                       |
                 |      ...                                                                             |
                 |          | <---------------- step_n ---------------> |                               |
                 |                  | <---------------------- eval --------------------> | <-- sync --> |
                 |--------------------------------------------------------------------------------------|
        trainer  | <-- idle --> | <-- step_1 --> | <-- step_2 --> | ... | <-- step_n --> | <-- sync --> |
        """
        while True:
            try:
                self.logger.info(f"Explore step {self.explore_step_num + 1} started.")
                explore_contionue = await self.explore_step()
                if not explore_contionue:
                    # TODO: support eval on last checkpoint
                    break
                if self.need_eval():
                    self.eval()
                if self.need_sync():
                    await self.sync_weight()
            except Exception:
                self.logger.error(f"Error in Explorer: {traceback.format_exc()}")
                break
        self.logger.info("--------------------\n> Explorer finished.\n--------------------")
        return self.config.explorer.name

    async def explore_step(self) -> bool:
        algo_config = self.algorithm_manager.get_current_algorithm_config(self.explore_step_num + 1)
        # skip warmup
        if algo_config.algorithm_type == "sft":
            self.explore_step_num += 1
            return True
        try:
            tasks = self.taskset.read()
        except StopIteration:
            self.logger.warning("No more tasks to explore. Stop exploring.")
            await self.save_checkpoint(sync_weight=False)
            self.status = RunningStatus.STOPPED
            await self.experience_buffer.release()
            return False
        self.scheduler.schedule(tasks, batch_id=self.explore_step_num + 1)
        self.explore_step_num += 1
        return True

    def need_sync(self) -> bool:
        if self.explore_step_num <= self.config.synchronizer.sync_offset:
            return False
        return (
            self.explore_step_num - self.config.synchronizer.sync_offset
        ) % self.config.synchronizer.sync_interval == 0

    def need_eval(self) -> bool:
        return self.explore_step_num % self.config.explorer.eval_interval == 0

    def eval(self):
        """Evaluation on all evaluation data samples."""
        if len(self.config.buffer.explorer_input.eval_tasksets) == 0:
            self.logger.warning("No evaluation data samples. Skip evaluation.")
            return
        self.logger.info(f"Evaluation at step {self.explore_step_num} started.")
        for eval_taskset_config in self.config.buffer.explorer_input.eval_tasksets:
            self.logger.info(
                f"Evaluation on {eval_taskset_config.name} at step {self.explore_step_num} started."
            )
            eval_taskset = get_buffer_reader(eval_taskset_config, self.config.buffer)
            eval_batch_id = f"{self.explore_step_num}/{eval_taskset.name}"
            self.pending_eval_tasks.append((self.explore_step_num, eval_taskset.name))
            while True:
                try:
                    self.scheduler.schedule(eval_taskset.read(), batch_id=eval_batch_id)
                except StopIteration:
                    break

    async def benchmark(self) -> bool:
        """Benchmark the model checkpoints."""
        # benchmark on the latest checkpoint
        if self.config.explorer.bench_on_latest_checkpoint:
            self.explore_step_num = await self._checkpoint_weights_update()
            self.eval()
            await self._finish_eval_step(prefix="bench")
            return True

        # benchmark on base model
        if self.config.explorer.eval_on_startup:
            await self._finish_eval_step(prefix="bench")

        # benchmark on all checkpoints
        all_ckp_steps = sorted(
            [
                int(ckp.split("global_step_")[-1])
                for ckp in os.listdir(self.config.checkpoint_job_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_job_dir, ckp))
                and ckp.startswith("global_step_")
            ]
        )
        for step_num in all_ckp_steps:
            self.explore_step_num = await self._checkpoint_weights_update(step_num=step_num)
            self.eval()
            await self._finish_eval_step(prefix="bench")
        return True

    async def save_checkpoint(self, sync_weight: bool = False) -> None:
        if not self.config.explorer.collect_experiences:
            # wait for all tasks to complete
            self.logger.info("Waiting for all tasks to complete")
            await self.scheduler.wait_all()
            self.logger.info(f"All tasks before step {self.explore_step_num} have completed.")
        log_task = asyncio.create_task(
            self._finish_steps(self.last_sync_step + 1, self.explore_step_num)
        )

        if sync_weight:
            # sync weights
            self.logger.info(f"Explorer sync_weights at step {self.explore_step_num} started.")
            if self.use_checkpoint_weights_update:
                await self._checkpoint_weights_update()
            else:  # nccl weights update
                await self._nccl_weights_update()
            self.last_sync_step = self.explore_step_num
            self.logger.info(f"Explorer sync_weights at step {self.explore_step_num} finished")

        # overlay log and weight sync
        await log_task

        # save explore checkpoint
        self.cache.save_explorer(
            current_step=self.explore_step_num,
            current_task_index=self.explore_step_num * self.config.buffer.batch_size,
        )

    async def sync_weight(self) -> None:
        """Synchronize model weights."""
        # call this method before training start to load the latest model weights
        await self.save_checkpoint(sync_weight=True)

    async def _finish_steps(self, start_step: int, end_step: int) -> None:
        for step in range(start_step, end_step + 1):
            self.logger.info(f"Log metrics of step {step}")
            await self._finish_explore_step(step=step)
            await self._finish_eval_step(step=step)

    async def _finish_explore_step(self, step: int) -> None:
        statuses, exps = await self.scheduler.get_results(batch_id=step)
        metric = {}
        if self.config.explorer.collect_experiences:
            exp_cnt, add_strategy_metric = await self.add_strategy.add(exps, step)
            self.generated_experience_cnt += exp_cnt
            metric.update(add_strategy_metric)
            metric["rollout/experience_count"] = exp_cnt
        if statuses:
            metric.update(gather_metrics([status.metric for status in statuses], "rollout"))
            self.monitor.log(metric, step=step)

    async def _finish_eval_step(self, step: Optional[int] = None, prefix: str = "eval") -> None:
        if not self.pending_eval_tasks:
            return
        step = step or self.explore_step_num
        st = time.time()
        metric = {}
        while self.pending_eval_tasks:
            eval_step, eval_task_name = self.pending_eval_tasks[0]
            if eval_step != step:
                return
            self.pending_eval_tasks.popleft()
            eval_results, _ = await self.scheduler.get_results(f"{step}/{eval_task_name}")
            metric.update(
                gather_metrics(
                    [status.metric for status in eval_results], f"{prefix}/{eval_task_name}"
                )
            )
        metric[f"{prefix}/total_time"] = time.time() - st
        self.monitor.log(metric, step)

    async def running_status(self) -> RunningStatus:
        return self.status

    async def shutdown(self) -> None:
        self.monitor.close()
        await self.scheduler.stop()
