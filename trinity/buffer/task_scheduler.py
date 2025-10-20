# -*- coding: utf-8 -*-
"""The taskset scheduler."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from trinity.buffer.buffer import get_buffer_reader
from trinity.common.config import Config

TASKSET_SCHEDULE_METRIC = "taskset_schedule"


@dataclass
class TaskIndex:
    taskset_id: Optional[int] = None
    index: Optional[int] = None


class TasksetScheduler:
    def __init__(self, explorer_state, config: Config):
        if "latest_task_index" in explorer_state:
            assert len(config.buffer.explorer_input.tasksets) == 1  # old format
            explorer_state["taskset"] = [
                {
                    "index": explorer_state["latest_task_index"],
                }
            ]

        self.read_batch_size = config.buffer.batch_size
        tasksets_config = config.buffer.explorer_input.tasksets

        tasksets_state = explorer_state.get("taskset", [{"index": 0}] * len(tasksets_config))
        self.tasksets = []
        for taskset_config, taskset_state in zip(tasksets_config, tasksets_state):
            taskset_config.index = taskset_state["index"]
            assert not taskset_config.is_eval
            taskset = get_buffer_reader(taskset_config, config.buffer)
            taskset.load_state_dict(taskset_state.get("state_dict", {}))
            self.tasksets.append(taskset)

        # self.step = explorer_state.get("latest_iteration", 0)  # TODO
        # total_epoch = config.buffer.total_epochs
        self.tasksets_queue = deque()
        for i, taskset in enumerate(self.tasksets):
            self.tasksets_queue.append((i, taskset))

    def read(self, batch_size: Optional[int] = None) -> List:
        batch_size = batch_size or self.read_batch_size
        batch = []
        for _ in range(len(self.tasksets_queue)):
            taskset_id, taskset = self.tasksets_queue.popleft()
            try:
                batch = taskset.read(batch_size)
                assert (
                    len(batch) == batch_size
                ), f"Batch size mismatch: {len(batch)} != {batch_size}"
                self.tasksets_queue.append((taskset_id, taskset))
                for task in batch:
                    task.index.taskset_id = taskset_id
                break
            except StopIteration:
                pass
        if len(batch) == 0:
            raise StopIteration
        return batch

    async def read_async(self, batch_size: Optional[int] = None) -> List:
        try:
            return self.read(batch_size)
        except StopIteration as e:
            raise StopAsyncIteration from e

    def state_dict(self) -> List[Dict]:
        return [
            {
                "index": taskset.index,
                "state_dict": taskset.state_dict(),
            }
            for taskset in self.tasksets
        ]

    def update(self, explore_metric: Dict) -> None:
        if TASKSET_SCHEDULE_METRIC not in explore_metric:
            return
        metric = explore_metric.pop(TASKSET_SCHEDULE_METRIC)
        taskset_update_kwargs = {}
        for index, value in metric.items():
            taskset_id = index.taskset_id
            if taskset_id not in taskset_update_kwargs:
                taskset_update_kwargs[taskset_id] = {
                    "indices": [],
                    "values": [],
                }
            kwargs = taskset_update_kwargs[taskset_id]
            kwargs["indices"].append(index.index)
            kwargs["values"].append(value)
        for taskset_id, kwargs in taskset_update_kwargs.items():
            taskset = self.tasksets[taskset_id]
            if hasattr(taskset, "update"):
                taskset.update(**kwargs)
