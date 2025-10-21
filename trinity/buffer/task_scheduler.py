# -*- coding: utf-8 -*-
"""The taskset scheduler."""

import copy
from collections import Counter
from typing import Dict, List

import numpy as np

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.selector import SELECTORS
from trinity.common.config import Config

TASKSET_SCHEDULE_METRIC = "taskset_schedule"


class TasksetScheduler:
    def __init__(self, explorer_state, config: Config):
        self.config = config
        if "latest_task_index" in explorer_state:
            assert len(config.buffer.explorer_input.tasksets) == 1  # old format
            explorer_state["taskset"] = [
                {
                    "current_index": explorer_state["latest_task_index"],
                }
            ]

        self.read_batch_size = config.buffer.batch_size
        tasksets_config = config.buffer.explorer_input.tasksets

        from trinity.buffer.reader.file_reader import TaskFileReader

        tasksets_state = explorer_state.get("taskset", [{"index": 0}] * len(tasksets_config))
        self.tasksets = []
        self.selectors = []
        for taskset_config, taskset_state in zip(tasksets_config, tasksets_state):
            taskset_config.index = taskset_state["current_index"]
            assert not taskset_config.is_eval  # assume drop last
            taskset = get_buffer_reader(taskset_config, config.buffer)
            assert isinstance(taskset, TaskFileReader), "Currently only support `TaskFileReader`"
            selector = SELECTORS.get(taskset_config.task_selector.selector_type)(
                taskset.dataset, taskset_config
            )
            selector.load_state_dict(taskset_state.get("state_dict", {}))
            self.tasksets.append(taskset)
            self.selectors.append(selector)

        # assume each explorer_step will only call read_async once
        self.step = explorer_state.get("latest_iteration", 0)
        self.base_taskset_ids = []
        for i, taskset in enumerate(self.tasksets):
            self.base_taskset_ids.extend([i] * len(taskset))
        self.steps_per_epoch = len(self.base_taskset_ids) // self.read_batch_size
        self.epoch = self.step // self.steps_per_epoch
        self.orders = self.build_orders(self.epoch)

    def build_orders(self, epoch: int):
        taskset_ids = copy.deepcopy(self.base_taskset_ids)
        rng = np.random.default_rng(epoch)
        rng.shuffle(taskset_ids)
        return taskset_ids

    async def read_async(self) -> List:
        batch_size = self.read_batch_size
        batch = []
        if self.step // self.steps_per_epoch != self.epoch:
            self.epoch = self.step // self.steps_per_epoch
            if self.epoch >= self.config.buffer.total_epochs:
                raise StopAsyncIteration
            self.orders = self.build_orders(self.epoch)
        start = (self.step - self.epoch * self.steps_per_epoch) * batch_size
        taskset_ids = self.orders[start : start + batch_size]
        counter = Counter(taskset_ids)
        for taskset_id, count in counter.items():
            indices = self.selectors[taskset_id].get_indices(batch_size=count)
            batch.extend(await self.tasksets[taskset_id].read_with_indices_async(indices))
        self.step += 1
        return batch

    def state_dict(self) -> List[Dict]:
        return [
            {
                "state_dict": selector.state_dict(),
            }
            for selector in self.selectors
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
