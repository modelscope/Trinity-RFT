# -*- coding: utf-8 -*-
"""
Trainer Class
"""
from __future__ import annotations

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
import ray

from trinity.algorithm import SAMPLE_STRATEGY
from trinity.common.config import Config
from trinity.common.constants import RunningStatus, SyncMethod, SyncStyle
from trinity.common.experience import Experiences
from trinity.manager.state_manager import StateManager
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR
from trinity.utils.plugin_loader import load_plugins
from trinity.utils.timer import Timer


class Trainer:
    """Consume the experience and train the model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(config.trainer.name, in_ray_actor=True)
        load_plugins()
        self.synchronizer = Synchronizer.get_actor(config)
        self.engine = get_trainer_wrapper(config)
        self.state = StateManager(
            path=config.checkpoint_job_dir, trainer_name=config.trainer.name, config=config
        )
        trainer_state = self.state.load_trainer()
        config.buffer.trainer_input.experience_buffer.index = trainer_state.get(
            "latest_exp_index", 0
        )
        self.last_trainer_sync_step = 0
        self.monitor = MONITOR.get(config.monitor.monitor_type)(
            project=config.project,
            group=self.config.group,
            name=config.name,
            role=config.trainer.name,
            config=config,
        )
        self._sample_exps_to_log = []
        self.sample_strategy = SAMPLE_STRATEGY.get(config.algorithm.sample_strategy)(
            buffer_config=config.buffer,
            **config.algorithm.sample_strategy_args,
        )
        self.save_interval = config.trainer.save_interval
        self.last_sync_step = None
        self.last_sync_time = None
        self.total_steps = config.trainer.total_steps or float("inf")

    async def prepare(self) -> None:
        """Prepare the trainer."""
        self.engine.prepare()
        self.last_trainer_sync_step = self.train_step_num
        await self.synchronizer.set_trainer_status.remote(RunningStatus.RUNNING)

    async def train(self) -> str:
        """Train the model."""
        while self.train_step_num < self.total_steps:
            try:
                metrics = {}
                # sample may be blocked due to explorer does not generate enough data
                self.logger.info(f"Sample data for step {self.train_step_num + 1} started.")
                sample_task = asyncio.create_task(self._sample_data())
                while not sample_task.done():
                    # sync weight to make sure the explorer can continue to explore and generate enough data
                    if await self.need_sync():
                        metrics.update(await self.sync_weight())
                    await asyncio.sleep(1)
                exps, sample_metrics, repr_samples = await sample_task
                metrics.update(sample_metrics)
                self.logger.info(f"Sample data for step {self.train_step_num + 1} finished.")
                metrics.update(await self.train_step(exps))
                if await self.need_sync():
                    metrics.update(await self.sync_weight())
                if self.need_save():
                    metrics.update(self.save_checkpoint())
                if self.config.trainer.enable_preview:
                    self._log_experiences(repr_samples)
                self.monitor.log(metrics, self.train_step_num)
            except StopAsyncIteration:
                self.logger.info("No more samples to train. Stopping training.")
                break
            except Exception:
                self.logger.error(f"Error in Trainer:\n{traceback.format_exc()}")
                break

        self.save_checkpoint(block_until_saved=True, save_as_hf=True)
        await self.synchronizer.set_trainer_status.remote(RunningStatus.STOPPED)
        self.logger.info("--------------------\n> Trainer finished.\n--------------------")
        return self.config.trainer.name

    async def train_step(self, exps: Experiences) -> Dict:
        """Train one step.

        Returns:
            bool: Whether to continue training.
            Dict: Metrics of the training step.
        """
        self.logger.info(f"Training at step {self.train_step_num + 1} started.")
        metrics = {}
        with Timer(metrics, "time/train_step"):
            train_metrics = self.engine.train_step(exps)
        self.logger.info(f"Training at step {self.train_step_num} finished.")
        metrics.update(train_metrics)
        return metrics

    async def _sample_data(self) -> Tuple[Experiences, Dict, List[Dict]]:
        """Sample a batch of experiences.

        Returns:
            Experiences: A batch of experiences.
            Dict: Metrics of the sampling step.
            List[Dict]: A list of representative samples for logging.
        """
        batch, metrics, repr_samples = await self.sample_strategy.sample(self.train_step_num + 1)
        metrics["sample/task_count"] = len(set(eid.tid for eid in batch.eids))
        return batch, metrics, repr_samples

    async def need_sync(self) -> bool:
        """Whether to sync the model weight."""
        if self.config.synchronizer.sync_style == SyncStyle.FIXED:
            return (
                self.last_sync_step != self.train_step_num
                and self.train_step_num % self.config.synchronizer.sync_interval == 0
            )
        else:
            if self.config.synchronizer.sync_style == SyncStyle.DYNAMIC_BY_TRAINER:
                delta = self.train_step_num - self.last_trainer_sync_step
                if delta >= self.config.synchronizer.sync_interval:
                    await self.synchronizer.set_trainer_status.remote(RunningStatus.REQUIRE_SYNC)
            explorer_status_counts = await self.synchronizer.get_explorer_status_counts.remote()
            if self.config.synchronizer.sync_method == SyncMethod.NCCL:
                return explorer_status_counts[RunningStatus.WAITING_SYNC] > 0
            else:  # memory & checkpoint
                return explorer_status_counts[RunningStatus.REQUIRE_SYNC] > 0

    def need_save(self) -> bool:
        """Whether to save the checkpoint."""
        return self.save_interval > 0 and self.train_step_num % self.save_interval == 0

    async def sync_weight(self) -> Dict:
        """Sync the model weight."""
        self.logger.info(f"Trainer sync_weights at step {self.train_step_num} started.")
        metrics = {}
        if self.last_sync_time is not None:
            metrics["time/trainer_sync_interval"] = time.time() - self.last_sync_time
        with Timer(metrics, "time/sync_weight"):
            if self.config.synchronizer.sync_method == SyncMethod.NCCL:
                result = await self.synchronizer.ready_to_nccl_sync.remote(
                    "trainer", self.train_step_num
                )
                if result is None:
                    self.logger.error("Trainer sync_weights failed.")
                else:
                    self.engine.sync_weight()
                    self.last_trainer_sync_step = self.train_step_num
            elif self.config.synchronizer.sync_method == SyncMethod.CHECKPOINT:
                self.engine.save_state_dict()
            elif self.config.synchronizer.sync_method == SyncMethod.MEMORY:
                self.engine.upload_state_dict()
            self.last_sync_step = self.train_step_num
            self.last_sync_time = time.time()
            await self.synchronizer.set_trainer_status.remote(RunningStatus.RUNNING)
        self.logger.info(f"Trainer sync_weights at step {self.train_step_num} finished.")
        return metrics

    def _log_experiences(self, samples: List[Dict]) -> None:
        self._sample_exps_to_log.extend(samples)
        if self.train_step_num % self.config.synchronizer.sync_interval == 0:
            self.monitor.log_table(
                "rollout_examples", pd.DataFrame(self._sample_exps_to_log), self.train_step_num
            )
            self._sample_exps_to_log.clear()

    def save_checkpoint(self, block_until_saved: bool = False, save_as_hf: bool = False) -> Dict:
        metrics = {}
        with Timer(metrics, "time/save_checkpoint"):
            self.logger.info(f"Saving checkpoint at step {self.train_step_num}...")
            self.engine.save_checkpoint(block_until_saved=block_until_saved, save_as_hf=save_as_hf)
            self.state.save_trainer(
                current_exp_index=self.engine.train_step_num * self.config.buffer.train_batch_size,
                current_step=self.train_step_num,
            )
        return metrics

    async def shutdown(self) -> None:
        self.monitor.close()

    @property
    def train_step_num(self) -> int:
        """Get the current training step number."""
        return self.engine.train_step_num

    async def is_alive(self) -> bool:
        """Check if the trainer is alive."""
        return True

    @classmethod
    def get_actor(cls, config: Config):
        """Get a Ray actor for the trainer."""
        return (
            ray.remote(cls)
            .options(name=config.trainer.name, namespace=ray.get_runtime_context().namespace)
            .remote(config)
        )


class TrainEngineWrapper(ABC):
    """A wrapper class to wrap various training engines."""

    @abstractmethod
    def prepare(self) -> None:
        """Do some preparation before training started."""

    @property
    @abstractmethod
    def train_step_num(self) -> int:
        """Get the current training step number."""

    @abstractmethod
    def train_step(self, batch: Experiences) -> Dict:
        """Training one step.

        Args:
            batch (Experiences): A batch of experiences to train.

        Returns:
            Dict: Metrics of the training step.
        """

    @abstractmethod
    def save_checkpoint(self, block_until_saved: bool = False, save_as_hf: bool = False) -> None:
        """Save the checkpoint."""

    @abstractmethod
    def sync_weight(self) -> None:
        """Sync the model weight."""

    @abstractmethod
    def upload_state_dict(self) -> None:
        """Upload the state dict to Synchronizer."""

    @abstractmethod
    def save_state_dict(self) -> None:
        """Only save the model state dict for Synchronizer."""


def get_trainer_wrapper(config: Config) -> TrainEngineWrapper:
    """Get a trainer wrapper."""
    if config.trainer.trainer_type == "verl":
        from trinity.trainer.verl_trainer import VerlPPOTrainerWrapper

        return VerlPPOTrainerWrapper(config)
    else:
        raise NotImplementedError
