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
    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        self.config = config
        self.trainer_status = RunningStatus.RUNNING
        self.last_trainer_sync_step = 0
        self.explorer_status_counter: Dict[RunningStatus, int] = {}
        self.last_explorer_sync_step = 0
        self._ready_condition = asyncio.Condition()
        self.model_state_dict = None
        self.model_version = 0
        self.checkpoint_shard_counter = defaultdict(lambda: 0)

    def set_trainer_status(self, status: RunningStatus):
        self.trainer_status = status

    def get_trainer_status(self) -> RunningStatus:
        return self.trainer_status

    def set_explorer_status(
        self, status: RunningStatus, old_status: Optional[RunningStatus] = None
    ):
        if old_status is not None:
            assert (
                old_status in self.explorer_status_counter
            ), f"Invalid explorer status {old_status}"
            assert old_status != status
            self.explorer_status_counter[old_status] -= 1
            assert self.explorer_status_counter[old_status] >= 0
        if status not in self.explorer_status_counter:
            self.explorer_status_counter[status] = 0
        self.explorer_status_counter[status] += 1

    def get_explorer_status_counter(self) -> Dict[RunningStatus, int]:
        return self.explorer_status_counter

    async def set_model_state_dict_with_step_num(
        self, step_num: Optional[int] = None, world_size: Optional[int] = None
    ) -> int:
        if world_size is not None:  # Used for trainer to update model
            assert step_num is not None
            self.checkpoint_shard_counter[step_num] += 1
            self.logger.info(
                f"Synchronizer received checkpoint {self.checkpoint_shard_counter[step_num]} of {world_size} shards"
            )
            if self.checkpoint_shard_counter[step_num] < world_size:
                return step_num

        checkpoint_dir, checkpoint_step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
            step_num=step_num,
        )
        model_state_dict = load_state_dict(os.path.join(checkpoint_dir, "actor"))  # TODO: to thread
        await self.set_model_state_dict(model_state_dict, checkpoint_step_num)
        return checkpoint_step_num

    async def set_model_state_dict(self, model_state_dict, trainer_step):
        self.model_state_dict = model_state_dict
        async with self._ready_condition:
            self.model_version = trainer_step
            self.logger.info(f"Set model state dict version to {trainer_step}.")
            self._ready_condition.notify_all()

    def get_model_state_dict(self):
        return self.model_state_dict, self.model_version

    def get_state_dict_meta(self):
        if self.model_state_dict is None:
            return None
        update_weight_args_list = []
        for name, param in self.model_state_dict.items():
            update_weight_args_list.append((name, str(param.dtype), tuple(param.shape)))
        return update_weight_args_list

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        explorer = ray.get_actor(self.config.explorer_name)
        await explorer.setup_weight_sync_group.remote(master_address, master_port, state_dict_meta)

    async def wait_new_model_state_dict(self, current_version: int) -> int:
        # wait for the new model state dict; return new version
        async with self._ready_condition:
            if self.model_version <= current_version:
                self.set_explorer_status(
                    RunningStatus.WAITING_SYNC, old_status=RunningStatus.REQUIRE_SYNC
                )
                await asyncio.wait_for(
                    self._ready_condition.wait(),
                    timeout=self.config.synchronizer.sync_timeout,
                )
            return self.model_version

    async def ready_to_nccl_sync(
        self, module: str, trainer_step: Optional[int] = None
    ) -> Union[int, None]:
        assert (
            sum(self.explorer_status_counter.values()) == 1
        ), "NCCL sync is only supported for one explorer."
        async with self._ready_condition:
            try:
                if module == "trainer":
                    self.model_version = trainer_step
                    self.trainer_status = RunningStatus.WAITING_SYNC
                    self._ready_condition.notify_all()
                    if self.explorer_status_counter[RunningStatus.WAITING_SYNC] != 1:
                        await asyncio.wait_for(
                            self._ready_condition.wait_for(
                                lambda: self.explorer_status_counter[RunningStatus.WAITING_SYNC]
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
                                lambda: self.trainer_status == RunningStatus.WAITING_SYNC,
                            ),
                            timeout=self.config.synchronizer.sync_timeout,
                        )
                return self.model_version
            except asyncio.TimeoutError:
                another_module = "Trainer" if module == "explorer" else "Explorer"
                self.logger.error(
                    f"{another_module} is not ready for model weight sync in {self.config.synchronizer.sync_timeout} seconds."
                )
                return None

    @classmethod
    def get_actor(cls, config: Optional[Config] = None, namespace: Optional[str] = None):
        if config is not None:
            return (
                ray.remote(cls)
                .options(name="synchronizer", namespace=config.ray_namespace, get_if_exists=True)
                .remote(config)
            )
        return ray.get_actor("synchronizer", namespace=namespace)
