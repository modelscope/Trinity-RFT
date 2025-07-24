# -*- coding: utf-8 -*-
"""
Trainer Class
"""
from __future__ import annotations

import traceback
from abc import ABC, abstractmethod

import ray

from trinity.common.config import Config
from trinity.common.constants import RunningStatus, SyncMethod, SyncStyle
from trinity.common.synchronizer import Synchronizer
from trinity.utils.log import get_logger


class Trainer:
    """Consume the experience and train the model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.synchronizer = Synchronizer.get_actor(config)
        self.engine = get_trainer_wrapper(config)
        self.last_trainer_sync_step = 0

    def prepare(self) -> None:
        """Prepare the trainer."""
        self.engine.prepare()
        self.last_trainer_sync_step = self.engine.train_step_num
        ray.get(self.synchronizer.set_trainer_status.remote(RunningStatus.RUNNING))

    def train(self) -> str:
        """Train the model."""
        while True:
            try:
                train_continue = self.train_step()
                if not train_continue:
                    break
                if self.need_sync():
                    self.sync_weight()
            except Exception:
                self.logger.error(f"Error in Trainer:\n{traceback.format_exc()}")
                break
        self.logger.info("--------------------\n> Trainer finished.\n--------------------")
        return self.config.trainer.name

    def train_step(self) -> bool:
        """Train one step.

        Returns:
            bool: Whether to continue training.
        """
        return self.engine.train_step()

    def need_sync(self) -> bool:
        """Whether to sync the model weight."""
        if self.config.synchronizer.sync_style == SyncStyle.FIXED:
            return self.engine.train_step_num % self.config.synchronizer.sync_interval == 0
        else:
            if self.config.synchronizer.sync_style == SyncStyle.DYNAMIC_BY_TRAINER:
                delta = self.engine.train_step_num - self.last_trainer_sync_step
                if delta >= self.config.synchronizer.sync_interval:
                    ray.get(self.synchronizer.set_trainer_status.remote(RunningStatus.REQUIRE_SYNC))
            explorer_status_counter = ray.get(
                self.synchronizer.get_explorer_status_counter.remote()
            )
            if self.config.synchronizer.sync_method == SyncMethod.NCCL:
                return explorer_status_counter[RunningStatus.WAITING_SYNC] > 0
            else:  # memory & checkpoint
                return explorer_status_counter[RunningStatus.REQUIRE_SYNC] > 0

    def sync_weight(self) -> None:
        """Sync the model weight."""
        self.logger.info(
            f"Trainer synchronizing weights at step {self.engine.train_step_num} starting.."
        )
        if self.config.synchronizer.sync_method == SyncMethod.NCCL:
            result = ray.get(
                self.synchronizer.ready_to_nccl_sync.remote("trainer", self.engine.train_step_num)
            )
            if result is None:
                self.logger.error("Trainer synchronizing weights failed.")
            else:
                self.engine.sync_weight()
                self.last_trainer_sync_step = self.engine.train_step_num
        elif self.config.synchronizer.sync_method == SyncMethod.CHECKPOINT:
            self.engine.save_state_dict()
        elif self.config.synchronizer.sync_method == SyncMethod.MEMORY:
            self.engine.upload_state_dict()
        self.logger.info(f"Trainer synchronizing weights at step {self.engine.train_step_num} end.")
        ray.get(self.synchronizer.set_trainer_status.remote(RunningStatus.RUNNING))

    def shutdown(self) -> None:
        ray.get(self.synchronizer.set_trainer_status.remote(RunningStatus.STOPPED))
        self.engine.save_checkpoint()
        self.engine.monitor.close()
        self.engine.shutdown()


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
    def train_step(self) -> bool:
        """Training."""

    @abstractmethod
    def save_checkpoint(self) -> None:
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

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the engine."""


def get_trainer_wrapper(config: Config) -> TrainEngineWrapper:
    """Get a trainer wrapper."""
    if config.trainer.trainer_type == "verl":
        from trinity.trainer.verl_trainer import VerlPPOTrainerWrapper

        return VerlPPOTrainerWrapper(config)
    else:
        raise NotImplementedError
