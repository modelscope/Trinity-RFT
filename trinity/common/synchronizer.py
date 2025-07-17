"""  """

import asyncio
from typing import List

import ray

from trinity.common.config import Config
from trinity.common.constants import RunningStatus


class Synchronizer:
    def __init__(self, config: Config):
        self.config = config
        self.trainer_status = RunningStatus.RUNNING
        self.last_trainer_sync_step = 0
        self.explorer_status = RunningStatus.RUNNING
        self.last_explorer_sync_step = 0
        self.ready_count = 0
        self._ready_condition = asyncio.Condition()

    def set_trainer_status(self, status: RunningStatus):
        self.trainer_status = status

    def set_explorer_status(self, status: RunningStatus):
        self.explorer_status = status

    def get_trainer_status(self) -> RunningStatus:
        return self.trainer_status

    def get_explorer_status(self) -> RunningStatus:
        return self.explorer_status

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        explorer = ray.get_actor(self.config.explorer_name)
        await explorer.setup_weight_sync_group.remote(master_address, master_port, state_dict_meta)

    async def ready_to_sync(self, module: str):
        async with self._ready_condition:
            try:
                if module == "trainer":
                    self.trainer_status = RunningStatus.WAITING_SYNC
                    self._ready_condition.notify_all()
                    if self.explorer_status != RunningStatus.WAITING_SYNC:
                        await asyncio.wait_for(
                            self._ready_condition.wait_for(
                                lambda: self.explorer_status == RunningStatus.WAITING_SYNC,
                            ),
                            timeout=self.config.synchronizer.sync_timeout,
                        )
                elif module == "explorer":
                    self.explorer_status = RunningStatus.WAITING_SYNC
                    self._ready_condition.notify_all()
                    if self.trainer_status != RunningStatus.WAITING_SYNC:
                        await asyncio.wait_for(
                            self._ready_condition.wait_for(
                                lambda: self.trainer_status == RunningStatus.WAITING_SYNC,
                            ),
                            timeout=self.config.synchronizer.sync_timeout,
                        )
                return True
            except asyncio.TimeoutError:
                another_module = "Trainer" if module == "explorer" else "Explorer"
                self.logger.error(
                    f"{another_module} is not ready for model weight sync in {self.config.synchronizer.sync_timeout} seconds."
                )
                return False

    @classmethod
    def get_actor(cls, config: Config):
        return (
            ray.remote(cls)
            .options(name="synchronizer", namespace=config.ray_namespace, get_if_exists=True)
            .remote(config)
        )
