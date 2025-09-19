import asyncio
from collections import deque
from typing import Dict, List, Set

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
        self.running_models: deque[int] = deque()
        self.requiring_sync_models: Set[int] = set()
        self.waiting_sync_models: Set[int] = set()
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

    async def schedule_weights_sync(self, model_version: int):
        if not self.running:
            self.logger.warning("Server is not running.")
            return

        while len(self.running_models) > self.explorer.config.explorer.min_running_model_num:
            if self.models[self.running_models[0]].model_version < self.latest_model_version:
                idx = self.running_models.popleft()
                self.requiring_sync_models.add(idx)
                self.models[idx].status = RunningStatus.REQUIRE_SYNC
                self.logger.info(f"Model {idx} scheduled for synchronization.")
                asyncio.create_task(self.models[idx].sync_model_weights(self.latest_model_version))

    async def _sync_model_weights(self, index: int):
        if self.models[index].status != RunningStatus.REQUIRE_SYNC:
            return
        current_load = await self.models[index].get_current_load()
        if current_load == 0:
            self.models[index].status = RunningStatus.WAITING_SYNC
            self.requiring_sync_models.remove(index)
            self.waiting_sync_models.add(index)
            self.logger.info(f"Model {index} begins synchronization.")
        else:
            self.logger.info(f"Model {index} still requires synchronization.")

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
        self.serve_task.cancel()
        try:
            await self.serve_task
        except asyncio.CancelledError:
            pass
        self.running = False
        self.logger.info("API server shut down.")
