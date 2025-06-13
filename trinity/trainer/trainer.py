# -*- coding: utf-8 -*-
"""
Trainer Class
This file is modified from verl.trainer.main_ppo.py
And is a reproduction code of Jiayi-Pan/TinyZero.

Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os
from abc import ABC, abstractmethod
from typing import Tuple

import ray

from trinity.algorithm.algorithm_manager import AlgorithmManager
from trinity.common.config import Config
from trinity.common.constants import SyncMethod
from trinity.utils.log import get_logger


@ray.remote(name="trainer")
class Trainer:
    """Consume the experience and train the model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.algorithm_manager = AlgorithmManager(config)
        self.engine = get_trainer_wrapper(config)

    def prepare(self) -> None:
        """Prepare the trainer."""
        self.engine.prepare()

    def train(self):
        """Train the model."""
        while True:
            train_status, _ = self.train_step()
            if not train_status:
                break

    def train_one_period(self) -> Tuple[bool, int]:
        """Train for one period. Each period contains `sync_interval` steps.
        Returns:
            train_status: Whether to continue training.
            train_step_num: The number of training steps"""
        for _ in range(self.config.synchronizer.sync_interval):
            train_status, train_step_num = self.train_step()
            if not train_status:
                return False, train_step_num
        self.logger.info(f"Train step {train_step_num} finished.")
        return True, train_step_num

    def train_step(self) -> Tuple[bool, int]:
        """Train one step.

        Returns:
            bool: Whether to continue training.
        """
        return self.engine.train_step()

    def sync_weight(self) -> None:
        """Sync the model weight."""
        if self.config.synchronizer.sync_method == SyncMethod.NCCL:
            self.engine.sync_weight()

    def flush_log(self, step: int) -> None:
        """Flush the log of the current step."""
        self.engine.logger.log({}, step=step, commit=True)

    def shutdown(self) -> None:
        # if checkpoint not saved, save the last checkpoint
        step_num = self.engine.train_step_num
        path = os.path.join(self.config.checkpoint_job_dir, f"global_step_{step_num}")
        if not os.path.isdir(path) or len(os.listdir(path)) == 0:
            self.engine.save_checkpoint()
        self.engine.logger.close()


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
    def train_step(self) -> Tuple[bool, int]:
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
