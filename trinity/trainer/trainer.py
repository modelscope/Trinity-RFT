# -*- coding: utf-8 -*-
"""
Trainer Class
This file is modified from verl.trainer.main_ppo.py
And is a reproduction code of Jiayi-Pan/TinyZero.

Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import ray

from trinity.buffer import get_buffer_reader
from trinity.common.config import Config
from trinity.common.constants import AlgorithmType, ReadStrategy, SyncMethod
from trinity.common.experience import Experiences
from trinity.utils.log import get_logger


@ray.remote(name="trainer")
class Trainer:
    """Consume the experience and train the model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.train_buffer = get_buffer_reader(
            self.config.buffer.trainer_input.experience_buffer,  # type: ignore
            self.config.buffer,
        )
        self.sft_warmup_buffer = (
            get_buffer_reader(
                self.config.buffer.trainer_input.sft_warmup_dataset,  # type: ignore
                self.config.buffer,
            )
            if self.config.trainer.sft_warmup_steps > 0
            else None
        )
        self.engine = get_trainer_wrapper(config)

    def prepare(self) -> None:
        """Prepare the trainer."""
        self.engine.prepare()

    def train(self, algo_type: AlgorithmType = AlgorithmType.PPO):
        """Train the model."""
        while True:
            train_status, _ = self.train_step(algo_type)
            if not train_status:
                break

    def train_one_period(self, algo_type: AlgorithmType = AlgorithmType.PPO) -> Tuple[bool, int]:
        """Train for one period. Each period contains `sync_interval` steps.
        Returns:
            train_status: Whether to continue training.
            train_step_num: The number of training steps"""
        for _ in range(self.config.synchronizer.sync_interval):
            train_status, train_step_num = self.train_step(algo_type)
            if not train_status:
                return False, train_step_num
        self.logger.info(f"Trainer steps {train_step_num} finished.")
        return True, train_step_num

    def train_step(self, algo_type: AlgorithmType = AlgorithmType.PPO) -> Tuple[bool, int]:
        """Train one step.

        Args:
            algo_type (AlgorithmType): The type of data to be used for training.
                Defaults to AlgorithmType.PPO.

        Returns:
            bool: Whether to continue training.
        """
        self.engine.set_mode(algo_type)
        if algo_type.is_rft() and self.config.trainer.get_exp_strategy:
            strategy = ReadStrategy(self.config.trainer.get_exp_strategy)
        else:
            strategy = None
        try:
            if algo_type.is_sft():
                exps = self.sft_warmup_buffer.read()
            else:
                exps = self.train_buffer.read(strategy=strategy)
        except StopIteration:
            self.logger.warning("No more data to train. Stop training.")
            return False, 0  # TODO: get the actual step number

        if algo_type.is_sft():
            return self.engine.train_sft_step(
                Experiences.gather_experiences(
                    exps,
                    pad_token_id=self.config.buffer.pad_token_id,  # type: ignore
                )
            )
        elif algo_type.is_rft():
            return self.engine.train_rft_step(
                Experiences.gather_experiences(
                    exps,
                    pad_token_id=self.config.buffer.pad_token_id,  # type: ignore
                )
            )
        elif algo_type.is_dpo():
            return self.engine.train_dpo_step(
                Experiences.gather_dpo_experiences(
                    exps,
                    pad_token_id=self.config.buffer.pad_token_id,  # type: ignore
                )
            )
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type}")

    def sync_weight(self) -> None:
        """Sync the model weight."""
        if self.config.synchronizer.sync_method == SyncMethod.NCCL:
            self.engine.sync_weight()

    def flush_log(self, step: int) -> None:
        """Flush the log of the current step."""
        self.engine.logger.log({}, step=step, commit=True)


class TrainEngineWrapper(ABC):
    """A wrapper class to wrap various training engines."""

    @abstractmethod
    def prepare(self) -> None:
        """Do some preparation before training started."""

    @abstractmethod
    def train_rft_step(self, experiences) -> Tuple[bool, int]:
        """Train on the RFT data."""

    @abstractmethod
    def train_sft_step(self, experiences) -> Tuple[bool, int]:
        """Train on the SFT data."""

    @abstractmethod
    def train_dpo_step(self, experiences) -> Tuple[bool, int]:
        """Train on the DPO data."""

    @abstractmethod
    def save_checkpoint(self) -> None:
        """Save the checkpoint."""

    @abstractmethod
    def sync_weight(self) -> None:
        """Sync the model weight."""

    @abstractmethod
    def set_mode(self, algo_type: AlgorithmType) -> None:
        """Set training mode."""

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
