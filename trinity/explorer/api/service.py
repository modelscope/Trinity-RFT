import asyncio
import time
from collections import deque
from typing import Dict, List

import torch

from trinity.common.constants import RunningStatus
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.explorer.explorer import Explorer
from trinity.utils.log import get_logger


class ExplorerService:
    def __init__(self, explorer: Explorer, listen_address: str = "localhost", port: int = 8010):
        self.logger = get_logger(__name__)
        self.explorer = explorer
        self.app = None
        self.port = port
        self.listen_address = listen_address
        self.running = False
        self.models: List[ModelWrapper] = [ModelWrapper(model) for model in explorer.models]
        self.min_running_model_num = explorer.config.explorer.min_running_model_num
        self.check_interval = explorer.config.explorer.service_status_check_interval
        self.max_timeout = explorer.config.explorer.max_timeout
        self.running_models: deque[int] = deque()  # indices of running models
        self.sync_task_map: Dict[asyncio.Future, int] = {}  # sync task -> model index
        self.latest_model_version = 0
        self.experience_queue = asyncio.Queue()
        self.experience_count = 0

    async def serve(self):
        from trinity.explorer.api.api import run_app

        if self.running:
            self.logger.warning("Server is already running.")
            return

        self.running = True
        await asyncio.gather(*[model.prepare() for model in self.models])

        for i, _ in enumerate(self.models):
            self.running_models.append(i)

        self.serve_task = asyncio.create_task(
            run_app(service=self, listen_address=self.listen_address, port=self.port)
        )
        self.sync_model_weights_task = asyncio.create_task(self.model_weights_sync_loop())

    async def model_weights_sync_loop(self):
        self.logger.info("Starting model weights synchronization loop.")
        while self.running:
            for idx in list(self.running_models):
                if (
                    len(self.running_models) > self.explorer.config.explorer.min_running_model_num
                    and self.models[idx].model_version < self.latest_model_version
                ):
                    self.running_models.remove(idx)
                    self.models[idx].status = RunningStatus.REQUIRE_SYNC
                    self.logger.info(f"Model {idx} scheduled for synchronization.")
                    future = asyncio.create_task(self._wait_for_sync_start(idx))
                    self.sync_task_map[future] = idx
                    future.add_done_callback(self._sync_model_weights)
            # wait half interval
            await asyncio.sleep(self.check_interval / 2)
        self.logger.info("Model weights synchronization loop stopped.")

    def set_latest_model_version(self, version: int) -> None:
        if version > self.latest_model_version:
            self.latest_model_version = version
            self.logger.info(f"Updated latest model version to {version}.")

    async def _wait_for_sync_start(self, index: int):
        start_time = time.time()
        while time.time() - start_time < self.max_timeout:
            current_load = await self.models[index].get_current_load()
            if current_load == 0:
                self.models[index].status = RunningStatus.WAITING_SYNC
                self.logger.info(f"Model {index} begins synchronization.")
                return
            else:
                await asyncio.sleep(2)
        raise asyncio.TimeoutError(
            f"Timeout waiting for model {index} to be free for synchronization. Current load: {current_load}"
        )

    async def _sync_model_weights(self, task: asyncio.Future):
        index = self.sync_task_map.pop(task)
        latest_version = self.latest_model_version  # capture the latest version
        if task.cancelled():
            self.logger.warning(f"Synchronization of model {index} was cancelled.")
        elif task.exception():
            self.logger.error(f"Error during synchronization of model {index}: {task.exception()}")
        else:
            await self.models[index].sync_model_weights(latest_version)
            self.logger.info(f"Model {index} synchronized to version {latest_version}.")
        self.running_models.append(index)
        self.models[index].status = RunningStatus.RUNNING

    async def allocate_model(self, increase_count: bool = True) -> str:
        model = self.models[self.running_models[0]]
        if increase_count:
            model.request_count += 1
        self.running_models.rotate(-1)
        return model.api_address

    def collect_metrics(self) -> Dict:
        metrics = {}
        for i, model in enumerate(self.models):
            metrics[f"rollout/model_{i}/total_request_count"] = model.request_count
            metrics[f"rollout/model_{i}/model_version"] = model.model_version
        metrics["rollout/total_experience_count"] = self.experience_count
        return metrics

    async def check_requiring_sync_models(self):
        if not self.running:
            self.logger.warning("Server is not running.")
            return
        await asyncio.gather(
            *[self._sync_model_weights(idx) for idx in list(self.requiring_sync_models)]
        )

    async def record_experience(self, response):
        experiences = []
        for choice in response["choices"]:
            exp = Experience(
                tokens=torch.cat(
                    (
                        torch.tensor(response["prompt_token_ids"], dtype=torch.int32),
                        torch.tensor(choice["token_ids"], dtype=torch.int32),
                    )
                ),
                logprobs=choice.get("logprobs", None),
                prompt_length=len(response["prompt_token_ids"]),
                response_text=choice.get("message", {}).get("content", ""),
            )
            experiences.append(exp)
        self.experience_count += len(experiences)
        for exp in experiences:
            await self.experience_queue.put(exp)

    async def get_all_experiences(self) -> List:
        experiences = []
        while not self.experience_queue.empty():
            experiences.append(await self.experience_queue.get())
        return experiences

    async def shutdown(self):
        if not self.running:
            self.logger.warning("Server is not running.")
            return
        self.sync_model_weights_task.cancel()
        self.serve_task.cancel()
        try:
            await self.serve_task
        except asyncio.CancelledError:
            pass
        self.running = False
        self.logger.info("API server shut down.")
