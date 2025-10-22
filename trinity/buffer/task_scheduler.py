# -*- coding: utf-8 -*-
"""The taskset scheduler."""

import copy
from collections import Counter
from typing import Dict, List

import numpy as np

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.selector import SELECTORS
from trinity.common.config import Config

SELECTOR_METRIC = "taskset_schedule"


class TasksetScheduler:
    def __init__(self, explorer_state, config: Config):
        self.config = config
        if "latest_task_index" in explorer_state:
            assert len(config.buffer.explorer_input.tasksets) == 1  # old format
            explorer_state["taskset_states"] = [
                {
                    "current_index": explorer_state["latest_task_index"],
                }
            ]

        self.read_batch_size = config.buffer.batch_size
        taskset_configs = config.buffer.explorer_input.tasksets

        from trinity.buffer.reader.file_reader import TaskFileReader

        taskset_states = explorer_state.get(
            "taskset_states", [{"current_index": 0}] * len(taskset_configs)
        )
        self.tasksets = []
        self.selectors = []
        for taskset_config, taskset_state in zip(taskset_configs, taskset_states):
            assert not taskset_config.is_eval  # assume drop last
            taskset = get_buffer_reader(taskset_config, config.buffer)
            assert isinstance(taskset, TaskFileReader), "Currently only support `TaskFileReader`"
            selector = SELECTORS.get(taskset_config.task_selector.selector_type)(
                taskset.dataset, taskset_config.task_selector
            )
            selector.load_state_dict(taskset_state)
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
            tasks = await self.tasksets[taskset_id].read_with_indices_async(indices)
            for task in tasks:
                task.index["taskset_id"] = taskset_id
            batch.extend(tasks)
        self.step += 1
        return batch

    def state_dict(self) -> List[Dict]:
        return [selector.state_dict() for selector in self.selectors]

    def update(self, pipeline_metrics: Dict) -> None:
        if SELECTOR_METRIC not in pipeline_metrics:
            return
        selector_metric = pipeline_metrics[SELECTOR_METRIC]
        for taskset_id, taskset_kwargs in selector_metric.items():
            selector = self.selectors[taskset_id]
            selector.update(**taskset_kwargs)
