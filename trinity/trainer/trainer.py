# -*- coding: utf-8 -*-
"""
Trainer Class
"""
from __future__ import annotations

import os
import traceback
from abc import ABC, abstractmethod

import ray

from trinity.algorithm import SAMPLE_STRATEGY
from trinity.common.config import Config
from trinity.common.constants import RunningStatus, SyncMethod
from trinity.utils.log import get_logger


class Trainer:
    """Consume the experience and train the model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.engine = get_trainer_wrapper(config)
        self.explorer_ref = None
        self.sample_strategy = SAMPLE_STRATEGY.get(config.algorithm.sample_strategy)(
            buffer_config=config.buffer,
            trainer_type=config.trainer.trainer_type,
            **config.algorithm.sample_strategy_args,
        )

    def prepare(self) -> None:
        """Prepare the trainer."""
        self.engine.prepare()

    def train(self) -> str:
        """Train the model."""
        while True:
            try:
                # sample experiences for train step
                try:
                    batch, sample_metrics, exp_samples = self.sample_strategy.sample(
                        self.engine.global_steps + 1,
                    )
                    successful_sampling = True
                except StopIteration:
                    print("No more data to train. Stop training.")
                    if (
                        self.engine.config.trainer.save_freq == 0
                        or self.engine.global_steps % self.engine.config.trainer.save_freq != 0
                    ):  # TODO: double-check this if-condition
                        self.engine.logger.info(f"Saving at step {self.engine.global_steps}.")
                        self.engine.save_checkpoint()
                        self.engine.logger.info(f"Saved at step {self.engine.global_steps}.")
                    successful_sampling = False
                # TODO: get rid of self.engine.global_steps/config/logger?

                # run train step
                if successful_sampling:
                    train_continue = self.engine.train_step(batch, sample_metrics, exp_samples)
                else:
                    train_continue = False

                if not train_continue:
                    break
                if self.need_sync():
                    self.sync_weight()
            except Exception:
                self.logger.error(f"Error in Trainer:\n{traceback.format_exc()}")
                break
        self.logger.info("--------------------\n> Trainer finished.\n--------------------")
        return self.config.trainer.name

    def need_sync(self) -> bool:
        """Whether to sync the model weight."""
        return self.engine.train_step_num % self.config.synchronizer.sync_interval == 0

    def sync_weight(self) -> None:
        """Sync the model weight."""
        if self.config.synchronizer.sync_method == SyncMethod.NCCL:
            self.logger.info(
                f"Trainer synchronizing weights at step {self.engine.train_step_num} starting.."
            )
            if self.explorer_ref is None:
                self.explorer_ref = ray.get_actor(self.config.explorer.name)
            explorer_status = ray.get(self.explorer_ref.running_status.remote())
            if explorer_status == RunningStatus.STOPPED:
                self.logger.warning("Explorer has already stopped. Skipping sync weight.")
                return
            ray.get(self.explorer_ref.ready_to_sync.remote())
            self.engine.sync_weight()
            self.logger.info(
                f"Trainer synchronizing weights at step {self.engine.train_step_num} end."
            )

    def shutdown(self) -> None:
        # if checkpoint not saved, save the last checkpoint
        step_num = self.engine.train_step_num
        path = os.path.join(self.config.checkpoint_job_dir, f"global_step_{step_num}")
        if not os.path.isdir(path) or len(os.listdir(path)) == 0:
            self.engine.save_checkpoint()
        self.engine.monitor.close()


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
    def train_step(self, batch, sample_metrics, exp_samples) -> bool:
        """Training."""

    @abstractmethod
    def save_checkpoint(self) -> None:
        """Save the checkpoint."""

    @abstractmethod
    def sync_weight(self) -> None:
        """Sync the model weight."""

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
