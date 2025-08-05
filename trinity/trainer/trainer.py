# -*- coding: utf-8 -*-
"""
Trainer Class
"""
from __future__ import annotations

import asyncio
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
import ray

from trinity.algorithm import SAMPLE_STRATEGY
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import Config
from trinity.common.constants import RunningStatus, SyncMethod, SyncStyle
from trinity.common.experience import Experiences
from trinity.common.synchronizer import Synchronizer
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR


class Trainer:
    """Consume the experience and train the model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.synchronizer = Synchronizer.get_actor(config)
        ray.get(self.synchronizer.acquire.remote())
        self.engine = get_trainer_wrapper(config)
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
        self.train_continue = True
        self.last_sync_step = None

    def prepare(self) -> None:
        """Prepare the trainer."""
        self.engine.prepare()
        self.last_trainer_sync_step = self.train_step_num
        ray.get(self.synchronizer.set_trainer_status.remote(RunningStatus.RUNNING))

    async def train(self) -> str:
        """Train the model."""
        while self.train_continue:
            try:
                train_task = asyncio.create_task(self.train_step())
                while not train_task.done():
                    if self.need_sync():
                        self.sync_weight()
                    await asyncio.sleep(1)
                self.train_continue &= await train_task
                if self.train_continue and self.need_sync():
                    self.sync_weight()
            except Exception:
                self.logger.error(f"Error in Trainer:\n{traceback.format_exc()}")
                self.train_continue = False

        self.engine.save_checkpoint(block_until_saved=True)
        await self.synchronizer.set_trainer_status.remote(RunningStatus.STOPPED)
        self.logger.info("--------------------\n> Trainer finished.\n--------------------")
        return self.config.trainer.name

    async def train_step(self) -> bool:
        """Train one step.

        Returns:
            bool: Whether to continue training.
        """
        self.logger.info(f"Training at step {self.train_step_num + 1} started.")
        try:
            batch, sample_metrics, repr_samples = await self.sample_strategy.sample(
                self.train_step_num + 1
            )
        except (StopIteration, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "StopIteration" not in str(e):
                raise
            self.logger.info("No more samples to train. Stopping training.")
            if (
                self.config.trainer.save_interval == 0
                or self.train_step_num % self.config.trainer.save_interval != 0
            ):
                self.logger.info(f"Saving at step {self.train_step_num}.")
                self.engine.save_checkpoint()
                self.logger.info(f"Saved at step {self.train_step_num}.")
            return False
        self.logger.info(f"Sampling at step {self.train_step_num + 1} done.")
        continue_run, metrics = self.engine.train_step(batch)
        self.logger.info(f"Training at step {self.train_step_num} finished.")
        prefix_metrics(sample_metrics, "sample", metrics)
        self.monitor.log(data=metrics, step=self.train_step_num)
        if self.config.trainer.enable_preview:
            self._log_experiences(repr_samples)
        return continue_run

    def need_sync(self) -> bool:
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
                    ray.get(self.synchronizer.set_trainer_status.remote(RunningStatus.REQUIRE_SYNC))
            explorer_status_counts = ray.get(self.synchronizer.get_explorer_status_counts.remote())
            if self.config.synchronizer.sync_method == SyncMethod.NCCL:
                return explorer_status_counts[RunningStatus.WAITING_SYNC] > 0
            else:  # memory & checkpoint
                return explorer_status_counts[RunningStatus.REQUIRE_SYNC] > 0

    def sync_weight(self) -> None:
        """Sync the model weight."""
        self.logger.info(f"Trainer synchronizing weights at step {self.train_step_num} starting..")
        if self.config.synchronizer.sync_method == SyncMethod.NCCL:
            result = ray.get(
                self.synchronizer.ready_to_nccl_sync.remote("trainer", self.train_step_num)
            )
            if result is None:
                self.logger.error("Trainer synchronizing weights failed.")
            else:
                self.engine.sync_weight()
                self.last_trainer_sync_step = self.train_step_num
        elif self.config.synchronizer.sync_method == SyncMethod.CHECKPOINT:
            self.engine.save_state_dict()
        elif self.config.synchronizer.sync_method == SyncMethod.MEMORY:
            self.engine.upload_state_dict()
        self.logger.info(f"Trainer synchronizing weights at step {self.train_step_num} end.")
        self.last_sync_step = self.train_step_num
        ray.get(self.synchronizer.set_trainer_status.remote(RunningStatus.RUNNING))

    def _log_experiences(self, samples: List[Dict]) -> None:
        self._sample_exps_to_log.extend(samples)
        if self.train_step_num % self.config.synchronizer.sync_interval == 0:
            self.monitor.log_table(
                "rollout_examples", pd.DataFrame(self._sample_exps_to_log), self.train_step_num
            )
            self._sample_exps_to_log.clear()

    async def shutdown(self) -> None:
        self.monitor.close()
        if await self.synchronizer.release.remote() == 0:
            ray.kill(self.synchronizer)
            self.logger.info("Synchronizer stopped.")

    @property
    def train_step_num(self) -> int:
        """Get the current training step number."""
        return self.engine.train_step_num


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
    def train_step(self, batch: Experiences) -> Tuple[bool, Dict]:
        """Training one step.

        Args:
            batch (Experiences): A batch of experiences to train.

        Returns:
            bool: Whether to continue training.
            Dict: Metrics of the training step.
        """

    @abstractmethod
    def save_checkpoint(self, block_until_saved: bool = False) -> None:
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
