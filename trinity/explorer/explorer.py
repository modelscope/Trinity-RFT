# -*- coding: utf-8 -*-
"""The explorer module"""
from __future__ import annotations

import asyncio
import os
import time
import traceback
from collections import deque
from typing import List, Optional

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.pipelines.experience_pipeline import ExperiencePipeline
from trinity.buffer.task_scheduler import TasksetScheduler
from trinity.common.config import Config
from trinity.common.constants import (
    ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
    RunningStatus,
    SyncMethod,
    SyncStyle,
)
from trinity.common.models import create_inference_models
from trinity.common.models.utils import get_checkpoint_dir_with_step_num
from trinity.explorer.scheduler import Scheduler
from trinity.manager.state_manager import StateManager
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.annotations import Experimental
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR, gather_metrics
from trinity.utils.plugin_loader import load_plugins


class Explorer:
    """Responsible for exploring the taskset."""

    def __init__(self, config: Config):
        self.logger = get_logger(config.explorer.name, in_ray_actor=True)
        load_plugins()
        self.state = StateManager(
            path=config.checkpoint_job_dir, explorer_name=config.explorer.name, config=config
        )
        explorer_state = self.state.load_explorer()
        self.explore_step_num = explorer_state.get("latest_iteration", 0)
        self.last_sync_step = self.explore_step_num if self.explore_step_num > 0 else -1
        self.synchronizer = Synchronizer.get_actor(config)
        self.config = config
        self.models, self.auxiliary_models = create_inference_models(config)
        self.experience_pipeline = self._init_experience_pipeline()
        self.taskset = (
            TasksetScheduler(explorer_state, config) if self.config.mode != "serve" else None
        )
        self.scheduler = None
        self.monitor = MONITOR.get(self.config.monitor.monitor_type)(
            project=self.config.project,
            group=self.config.group,
            name=self.config.name,
            role=self.config.explorer.name,
            config=config,
        )
        self.batch_size = config.buffer.batch_size
        self.update_interval = (
            self.config.synchronizer.sync_interval * self.config.buffer.batch_size
        )
        self.use_nccl_sync = self.config.synchronizer.sync_method == SyncMethod.NCCL
        self.pending_eval_tasks = deque()

        # For checkpoint weights update
        # Use explorer to periodically load the latest model weights and
        # boradcast to all rollout models
        self.enable_lora = self.config.explorer.rollout_model.enable_lora
        self.model_version = -1
        self.last_sync_successful = True
        self.logger.info("Finished initializing Explorer.")

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        # In checkpoint mode, we use explorer to store the model weights which has no rank
        base_offset = 1 if self.use_nccl_sync else 0
        world_size = (
            len(self.models) * self.config.explorer.rollout_model.tensor_parallel_size + base_offset
        )
        self.logger.info(
            f"Initialize process group for weight synchronization, "
            f"master_address={master_address}, master_port={master_port}, "
            f"world_size={world_size}, rank_offset={base_offset}"
        )
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
                state_dict_meta=state_dict_meta,
            )
            for i, model in enumerate(self.models)
        ]
        await asyncio.gather(*refs)

    async def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> int:
        step_num = await self.synchronizer.set_model_state_dict_with_step_num.remote(step_num)
        await asyncio.gather(*[model.sync_model.remote(step_num) for model in self.models])
        return step_num  # type: ignore

    async def _pull_latest_weights(self):
        self.logger.info("Start to pull latest model weights.")
        new_version = await self.synchronizer.wait_new_model_state_dict.remote(self.model_version)
        if new_version > self.model_version:
            if self.model_version != -1:
                self.logger.info(f"New model weights version: {new_version}")
                await asyncio.gather(
                    *[model.sync_model.remote(new_version) for model in self.models]
                )
            self.model_version = new_version
            self.last_sync_step = self.explore_step_num
            self.last_sync_successful = True
        else:
            self.logger.warning(
                f"No new model weights found, current version: {self.model_version}"
            )
            self.last_sync_successful = False

    async def _nccl_weights_update(self):
        new_version = await self.synchronizer.ready_to_nccl_sync.remote(
            "explorer", self.model_version
        )
        if new_version is None:
            self.logger.info("Trainer is not ready to sync weight. Skipping sync weight.")
            self.last_sync_successful = False
            return
        self.model_version = new_version
        await asyncio.gather(
            *[model.sync_model.remote(self.model_version) for model in self.models]
        )
        self.last_sync_step = self.explore_step_num
        self.last_sync_successful = True

    async def prepare(self) -> None:
        """Preparation before running."""
        try:
            await self.experience_pipeline.prepare.remote()
            self.logger.info("Experience pipeline is ready.")
            # make sure all rollout models are ready
            model_ready_ref = [model.__ray_ready__.remote() for model in self.models]
            await asyncio.gather(*model_ready_ref)
            self.logger.info("All rollout models are ready.")

            if not self.use_nccl_sync:
                master_address, master_port = await self.models[0].get_available_address.remote()
                await self.setup_weight_sync_group(master_address, master_port)
            if self.config.mode != "serve":
                self.scheduler = Scheduler(self.config, self.models, self.auxiliary_models)
                await self.scheduler.start()
            if self.config.explorer.eval_on_startup and self.explore_step_num == 0:
                await self.eval()

            await self.synchronizer.set_explorer_status.remote(RunningStatus.REQUIRE_SYNC)
        except Exception as e:
            self.logger.error(f"Error during explorer preparation: {traceback.format_exc()}")
            await self.shutdown()
            raise e

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
                    await self.eval()
                if await self.need_sync():
                    await self.sync_weight()
            except Exception:
                self.logger.error(f"Error in Explorer: {traceback.format_exc()}")
                break
        self.logger.info(
            f"--------------------\n> Explorer ({self.config.explorer.name}) finished.\n--------------------"
        )
        return self.config.explorer.name

    async def explore_step(self) -> bool:
        try:
            tasks = await self.taskset.read_async()
        except StopAsyncIteration:
            self.logger.warning("No more tasks to explore. Stop exploring.")
            await self.save_checkpoint(sync_weight=False)
            await self.synchronizer.set_explorer_status.remote(
                RunningStatus.STOPPED,
                old_status=RunningStatus.RUNNING
                if self.last_sync_successful
                else RunningStatus.REQUIRE_SYNC,
            )
            await self.shutdown()
            return False
        self.scheduler.schedule(tasks, batch_id=self.explore_step_num + 1)
        self.explore_step_num += 1
        return True

    async def need_sync(self) -> bool:
        if self.config.synchronizer.sync_style == SyncStyle.FIXED:
            if self.explore_step_num <= self.config.synchronizer.sync_offset:
                return False
            require_sync = (
                self.explore_step_num - self.config.synchronizer.sync_offset
            ) % self.config.synchronizer.sync_interval == 0
        else:
            require_sync = False
            if self.config.synchronizer.sync_style == SyncStyle.DYNAMIC_BY_EXPLORER:
                delta = self.explore_step_num - self.last_sync_step
                if delta >= self.config.synchronizer.sync_interval:
                    require_sync = True
            else:
                require_sync = await (
                    self.synchronizer.get_trainer_status.remote() == RunningStatus.REQUIRE_SYNC
                )
        if require_sync and self.last_sync_successful:
            await self.synchronizer.set_explorer_status.remote(
                RunningStatus.REQUIRE_SYNC, old_status=RunningStatus.RUNNING
            )
        return require_sync

    def need_eval(self) -> bool:
        return self.explore_step_num % self.config.explorer.eval_interval == 0

    async def eval(self):
        """Evaluation on all evaluation data samples."""
        if len(self.config.buffer.explorer_input.eval_tasksets) == 0:
            self.logger.warning("No evaluation data samples. Skip evaluation.")
            return
        self.logger.info(f"Evaluation at step {self.explore_step_num} started.")

        if self.config.buffer.explorer_input.default_eval_workflow_type:
            self.logger.info(
                f"Use '{self.config.buffer.explorer_input.default_eval_workflow_type}' for evaluation."
            )

        for eval_taskset_config in self.config.buffer.explorer_input.eval_tasksets:
            self.logger.info(
                f"Evaluation on {eval_taskset_config.name} at step {self.explore_step_num} started."
            )
            eval_taskset = get_buffer_reader(eval_taskset_config)
            eval_batch_id = f"{self.explore_step_num}/{eval_taskset.name}"
            self.pending_eval_tasks.append((self.explore_step_num, eval_taskset.name))
            while True:
                try:
                    data = await eval_taskset.read_async()
                    self.scheduler.schedule(data, batch_id=eval_batch_id)
                except StopAsyncIteration:
                    break

    async def benchmark(self) -> bool:
        """Benchmark the model checkpoints."""
        # benchmark on the latest checkpoint
        if self.config.explorer.bench_on_latest_checkpoint:
            self.explore_step_num = await self._checkpoint_weights_update()
            await self.eval()
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
            await self.eval()
            await self._finish_eval_step(prefix="bench")
        return True

    async def save_checkpoint(self, sync_weight: bool = False) -> None:
        if self.scheduler:
            await self._finish_steps(
                self.last_sync_step + 1, self.explore_step_num, self.model_version
            )

        if sync_weight:
            # sync weights
            self.logger.info(f"Explorer sync_weights at step {self.explore_step_num} started.")
            if self.use_nccl_sync:
                await self._nccl_weights_update()
            else:  # pull weights from Synchronizer
                await self._pull_latest_weights()
            self.logger.info(
                f"Explorer sync_weights at step {self.explore_step_num} finished, model version = {self.model_version}."
            )

        # save explore checkpoint
        self.state.save_explorer(
            current_step=self.explore_step_num,
            taskset_states=self.taskset.state_dict() if self.taskset else [],
        )

    async def sync_weight(self) -> None:
        """Synchronize model weights."""
        # call this method before training start to load the latest model weights
        await self.save_checkpoint(sync_weight=True)

    async def _finish_steps(self, start_step: int, end_step: int, model_version: int) -> None:
        for step in range(start_step, end_step + 1):
            self.logger.info(f"Log metrics of step {step}")
            await self._finish_explore_step(step=step, model_version=model_version)
            await self._finish_eval_step(step=step)

    async def _finish_explore_step(self, step: int, model_version: int) -> None:
        statuses, exps = await self.scheduler.get_results(batch_id=step)
        metric = {"rollout/model_version": model_version}
        pipeline_metrics = await self.experience_pipeline.process.remote(exps)
        self.taskset.update(pipeline_metrics)
        metric.update(pipeline_metrics)
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

    async def shutdown(self) -> None:
        if self.scheduler:
            await self.scheduler.stop()
            self.scheduler = None
        if self.experience_pipeline:
            await self.experience_pipeline.close.remote()
            self.experience_pipeline = None
        if self.monitor:
            self.monitor.close()
            self.monitor = None
        self.logger.info(
            f"Explorer ({self.config.explorer.name}) shutdown successfully at step {self.explore_step_num}."
        )

    async def is_alive(self) -> bool:
        """Check if the explorer is alive."""
        return True

    def _init_experience_pipeline(self) -> ray.actor.ActorHandle:
        """Init experience pipeline for the explorer."""
        node_id = ray.get_runtime_context().get_node_id()
        return (
            ray.remote(ExperiencePipeline)
            .options(
                name=f"{self.config.explorer.name}_pipeline",
                namespace=ray.get_runtime_context().namespace,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            )
            .remote(self.config)
        )

    @Experimental
    async def serve(self) -> None:
        """Run the explorer in serving mode.

        In serving mode, the explorer starts an OpenAI compatible server to handle requests.
        Agent applications can be deployed separately and interact with the explorer via the API.


        .. code-block:: python

            import openai


            client = openai.OpenAI(
                base_url=f"{explorer_server_url}/v1",
                api_key="EMPTY",
            )
            response = client.chat.completions.create(
                model=config.model.model_path,
                messages=[{"role": "user", "content": "Hello!"}]
            )
        """
        from trinity.explorer.api.service import ExplorerService

        self.service = ExplorerService(
            self,
            listen_address=self.config.explorer.listen_address,
            port=self.config.explorer.api_port,
        )
        await self.service.serve()
        self.server_url = f"http://{ray.util.get_node_ip_address()}:{self.service.port}"
        self.logger.info(
            f"Explorer API Server is started on {self.server_url} and listening to {self.service.listen_address}."
        )
        self.state.save_explorer_server_url(self.server_url)
        while True:
            self.explore_step_num += 1
            await asyncio.sleep(self.config.explorer.service_status_check_interval)
            # process experiences generated in the last interval
            exps = await self.service.get_all_experiences()
            metrics = await self.experience_pipeline.process.remote(exps)
            metrics.update(self.service.collect_metrics())
            self.monitor.log(metrics, self.explore_step_num)
            # get the latest checkpoint
            _, step_num = get_checkpoint_dir_with_step_num(
                self.config.checkpoint_job_dir, raise_error=False
            )
            self.service.set_latest_model_version(step_num)

    @classmethod
    def get_actor(cls, config: Config):
        """Get a Ray actor for the explorer."""
        return (
            ray.remote(cls)
            .options(
                name=config.explorer.name,
                namespace=ray.get_runtime_context().namespace,
            )
            .remote(config)
        )
