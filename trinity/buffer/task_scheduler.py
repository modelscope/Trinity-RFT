# -*- coding: utf-8 -*-
"""The taskset scheduler."""

from collections import deque
from typing import Dict, List, Optional
from trinity.buffer.buffer import get_buffer_reader
from trinity.common.config import Config


class TasksetScheduler:
    def __init__(self, explorer_state, config: Config):
        if 'latest_task_index' in explorer_state:
            assert len(config.buffer.explorer_input.taskset) == 1  # old format
            explorer_state['taskset'] = [
                {
                    "index": explorer_state['latest_task_index'],
                }
            ]

        tasksets_config = config.buffer.explorer_input.tasksets

        tasksets_state = explorer_state.get('taskset', [{"index": 0}] * len(tasksets_config))
        self.tasksets = []
        for taskset_config, taskset_state in zip(tasksets_config, tasksets_state):
            taskset_config.index = taskset_state["index"]
            assert not taskset_config.is_eval
            self.tasksets.append(get_buffer_reader(taskset_config, config.buffer))
        self.tasksets_queue = deque()
        for taskset in self.tasksets:
            self.tasksets_queue.append(taskset)

    def read(self, batch_size: Optional[int] = None) -> List:
        batch = []
        for _ in range(len(self.tasksets_queue)):
            taskset = self.tasksets_queue.popleft()
            try:
                batch = taskset.read(batch_size)
                assert len(batch) == batch_size
                self.tasksets_queue.append(taskset)
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

    def save_state(self) -> Dict:
        return [
            {
                "index": taskset.index,
            }
            for taskset in self.tasksets
        ]

    def update(self, experiences, explore_metric, eval_metric) -> None:
        pass
