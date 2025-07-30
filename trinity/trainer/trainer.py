# -*- coding: utf-8 -*-
"""
Trainer Class
"""
from __future__ import annotations

import os
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
import ray

from trinity.algorithm import SAMPLE_STRATEGY
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import Config
from trinity.common.constants import RunningStatus, SyncMethod
from trinity.common.experience import Experiences
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR


class Trainer:
    """Consume the experience and train the model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.engine = get_trainer_wrapper(config)
        self.explorer_ref = None
        self.monitor = MONITOR.get(config.monitor.monitor_type)(
            project=config.project,
            name=config.name,
            role=config.trainer.name,
            config=config,
        )
        self._sample_exps_to_log = []
        self.sample_strategy = SAMPLE_STRATEGY.get(config.algorithm.sample_strategy)(
            buffer_config=config.buffer,
            **config.algorithm.sample_strategy_args,
        )

    def prepare(self) -> None:
        """Prepare the trainer."""
        self.engine.prepare()

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
        try:
            batch, sample_metrics, repr_samples = self.sample_strategy.sample(
                self.train_step_num + 1
            )
        except StopIteration:
            self.logger.info("No more samples to train. Stopping training.")
            if (
                self.config.trainer.save_interval == 0
                or self.train_step_num % self.config.trainer.save_interval != 0
            ):
                self.logger.info(f"Saving at step {self.train_step_num}.")
                self.engine.save_checkpoint()
                self.logger.info(f"Saved at step {self.train_step_num}.")
            return False
        continue_run, metrics = self.engine.train_step(batch)
        prefix_metrics(sample_metrics, "sample", metrics)
        self.monitor.log(data=metrics, step=self.train_step_num)
        if self.config.trainer.enable_preview:
            self._log_experiences(repr_samples)
        return continue_run

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

    def _log_experiences(self, samples: List[Dict]) -> None:
        self._sample_exps_to_log.extend(samples)
        if self.train_step_num % self.config.synchronizer.sync_interval == 0:
            self.monitor.log_table(
                "rollout_examples", pd.DataFrame(self._sample_exps_to_log), self.train_step_num
            )
            self._sample_exps_to_log.clear()

    def shutdown(self) -> None:
        # if checkpoint not saved, save the last checkpoint
        path = os.path.join(self.config.checkpoint_job_dir, f"global_step_{self.train_step_num}")
        if not os.path.isdir(path) or len(os.listdir(path)) == 0:
            self.engine.save_checkpoint()

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
