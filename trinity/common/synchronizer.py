"""A centralized synchronizer for coordinating explorer and trainer."""

import asyncio
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import ray

from trinity.common.config import Config
from trinity.common.constants import RunningStatus
from trinity.common.models.utils import (
    get_checkpoint_dir_with_step_num,
    load_state_dict,
)
from trinity.utils.log import get_logger


class Synchronizer:
    """
    A central component to manage synchronization of models and states between
    the trainer and one or more explorers in a distributed training setup.

    Attributes:
        trainer_status: Current status of the trainer (e.g., running, waiting).
        explorer_status_counts: Dictionary tracking the number of explorers in each status.
        _ready_condition: Async condition variable for signaling state changes.
        model_state_dict: The latest model weights.
        model_version: Version number of the current model.
        checkpoint_shard_counter: Tracks how many shards are received from trainer for a specific train step.
    """

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.config = config
        self.trainer_status = RunningStatus.STOPPED
        self.explorer_status_counts: Dict[RunningStatus, int] = defaultdict(lambda: 0)
        self._ready_condition = asyncio.Condition()
        self.model_state_dict = None
        self.model_version = 0
        self.checkpoint_shard_counter = defaultdict(lambda: 0)
        self.ref_count = 0

    async def set_trainer_status(self, status: RunningStatus):
        """Update the status of the trainer."""
        async with self._ready_condition:
            self.trainer_status = status
            if status == RunningStatus.STOPPED:
                self._ready_condition.notify_all()

    def get_trainer_status(self) -> RunningStatus:
        """Get the current status of the trainer."""
        return self.trainer_status

    def set_explorer_status(
        self, status: RunningStatus, old_status: Optional[RunningStatus] = None
    ):
        """
        Update the status count for an explorer.

        Args:
            status: New status of the explorer.
            old_status: Previous status if changing from one to another.
        """
        if old_status is not None:
            assert (
                old_status in self.explorer_status_counts
            ), f"Invalid explorer status {old_status}"
            assert old_status != status, f"Invalid status change from {old_status} to {status}"
            self.explorer_status_counts[old_status] -= 1
            assert (
                self.explorer_status_counts[old_status] >= 0
            ), f"Invalid status count {old_status} (new status {status})"
        if status not in self.explorer_status_counts:
            self.explorer_status_counts[status] = 0
        self.explorer_status_counts[status] += 1

    def get_explorer_status_counts(self) -> Dict[RunningStatus, int]:
        """Return the current status counts for all explorers."""
        return self.explorer_status_counts

    async def set_model_state_dict_with_step_num(
        self, step_num: Optional[int] = None, world_size: Optional[int] = None
    ) -> int:
        """
        Load and set the model state dictionary from a checkpoint at a specific step.

        Args:
            step_num: Training step number corresponding to the checkpoint.
            world_size: Number of shards expected for this checkpoint.

        Returns:
            The updated model version (step number).
        """
        if world_size is not None:  # Used when trainer updates the model
            assert step_num is not None
            assert self.checkpoint_shard_counter[step_num] < world_size, "World size mismatch!"
            self.checkpoint_shard_counter[step_num] += 1
            self.logger.info(
                f"Synchronizer has received {self.checkpoint_shard_counter[step_num]} out of {world_size} shards from the checkpoint {step_num}."
            )
            if self.checkpoint_shard_counter[step_num] < world_size:
                return step_num

        checkpoint_dir, checkpoint_step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
            step_num=step_num,
        )
        if checkpoint_step_num != self.model_version:
            model_state_dict = load_state_dict(os.path.join(checkpoint_dir, "actor"))
            await self.set_model_state_dict(model_state_dict, checkpoint_step_num)
        return checkpoint_step_num

    async def set_model_state_dict(self, model_state_dict: Union[dict, None], trainer_step: int):
        """
        Set the new model state and update the version.

        Args:
            model_state_dict: The PyTorch model state dictionary.
            trainer_step: Step number associated with this model version.
        """
        self.model_state_dict = model_state_dict
        async with self._ready_condition:
            self.model_version = trainer_step
            self.logger.info(f"Set model state dict version to {trainer_step}.")
            self._ready_condition.notify_all()

    def get_model_state_dict(self):
        """Return the current model state and its version."""
        return self.model_state_dict, self.model_version

    def get_state_dict_meta(self):
        """
        Return metadata about the model state (names, data types, shapes).

        Returns:
            List of tuples: (name, dtype, shape).
        """
        if self.model_state_dict is None:
            return None
        update_weight_args_list = []
        for name, param in self.model_state_dict.items():
            update_weight_args_list.append((name, str(param.dtype), tuple(param.shape)))
        return update_weight_args_list

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        """
        Notify the explorer actor to setup weight sync group.

        This is used to initialize NCCL-based synchronization for distributed training.

        Args:
            master_address: IP address of the master node.
            master_port: Port used for synchronization.
            state_dict_meta: Metadata of the model parameters.
        """
        explorer = ray.get_actor(self.config.explorer.name, namespace=self.config.ray_namespace)
        await explorer.setup_weight_sync_group.remote(master_address, master_port, state_dict_meta)

    async def wait_new_model_state_dict(self, current_version: int, no_wait: bool = False) -> int:
        """
        Wait until a new model state is available.

        Args:
            current_version: Current model version known to one explorer.

        Returns:
            The new model version after it has been updated.
        """
        async with self._ready_condition:
            assert (
                self.model_version >= current_version
            ), f"The model version in Synchronizer ({self.model_version}) should be no smaller than that in Explorer ({current_version})!"
            if self.model_version == current_version:
                if not no_wait and self.trainer_status != RunningStatus.STOPPED:
                    # TODO: explorer need support no wait
                    # TODO: handle timeout
                    await asyncio.wait_for(
                        self._ready_condition.wait(),
                        timeout=self.config.synchronizer.sync_timeout,
                    )
            if self.model_version > current_version:
                self.set_explorer_status(
                    RunningStatus.WAITING_SYNC, old_status=RunningStatus.REQUIRE_SYNC
                )
            return self.model_version

    async def ready_to_nccl_sync(
        self, module: str, trainer_step: Optional[int] = None
    ) -> Union[int, None]:
        """
        Prepare for NCCL-based synchronization between modules.

        Only supports one explorer currently.

        Args:
            module: Either 'trainer' or 'explorer'.
            trainer_step: Optional step number from the trainer.

        Returns:
            The model version if both sides are ready; otherwise None.
        """
        assert (
            sum(self.explorer_status_counts.values()) == 1
        ), "NCCL sync is only supported for one explorer."

        def sync_failed():
            if module == "explorer":
                another_module = "Trainer"
                self.set_explorer_status(
                    RunningStatus.REQUIRE_SYNC, old_status=RunningStatus.WAITING_SYNC
                )
            else:
                another_module = "Explorer"
                self.trainer_status = RunningStatus.REQUIRE_SYNC
            self.logger.error(f"{another_module} is not ready for model weight sync.")
            return None

        non_stop_cnt = sum(
            value
            for key, value in self.explorer_status_counts.items()
            if key != RunningStatus.STOPPED
        )
        if non_stop_cnt == 0:
            return sync_failed()

        async with self._ready_condition:
            try:
                if module == "trainer":
                    self.model_version = trainer_step
                    self.trainer_status = RunningStatus.WAITING_SYNC
                    self._ready_condition.notify_all()
                    if self.explorer_status_counts[RunningStatus.WAITING_SYNC] != 1:
                        await asyncio.wait_for(
                            self._ready_condition.wait_for(
                                lambda: self.explorer_status_counts[RunningStatus.WAITING_SYNC]
                                == 1,
                            ),
                            timeout=self.config.synchronizer.sync_timeout,
                        )
                elif module == "explorer":
                    self.set_explorer_status(
                        RunningStatus.WAITING_SYNC, old_status=RunningStatus.REQUIRE_SYNC
                    )
                    self._ready_condition.notify_all()
                    if self.trainer_status != RunningStatus.WAITING_SYNC:
                        await asyncio.wait_for(
                            self._ready_condition.wait_for(
                                lambda: self.trainer_status
                                in {RunningStatus.WAITING_SYNC, RunningStatus.STOPPED},
                            ),
                            timeout=self.config.synchronizer.sync_timeout,
                        )
                        if self.trainer_status == RunningStatus.STOPPED:
                            return sync_failed()
                    self.trainer_status = RunningStatus.RUNNING
                return self.model_version
            except asyncio.TimeoutError:
                return sync_failed()

    @classmethod
    def get_actor(cls, config: Optional[Config] = None, namespace: Optional[str] = None):
        """
        Get or create a remote Ray actor for the Synchronizer.

        Args:
            config: Optional configuration to use for creating the actor.
            namespace: Optional Ray namespace for the actor.

        Returns:
            A reference to the Synchronizer actor.
        """
        if config is not None:
            return (
                ray.remote(cls)
                .options(
                    name="synchronizer",
                    namespace=config.ray_namespace,
                    get_if_exists=True,
                    lifetime="detached",
                )
                .remote(config)
            )
        return ray.get_actor("synchronizer", namespace=namespace)

    def acquire(self):
        self.ref_count += 1

    def release(self):
        self.ref_count -= 1
        return self.ref_count
