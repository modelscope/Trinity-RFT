# -*- coding: utf-8 -*-
import copy
import random
from typing import Dict, List, Tuple

import numpy as np

from trinity.algorithm.add_strategy.add_strategy import (
    ADD_STRATEGY,
    AddStrategy,
    group_by,
)
from trinity.buffer import BufferWriter
from trinity.common.experience import Experience
from trinity.utils.timer import Timer


@ADD_STRATEGY.register_module("duplicate_informative")
class DuplicateInformativeAddStrategy(AddStrategy):
    """An AddStrategy that filters experiences based on reward variance and duplicates them to reach the target size.
    Ref: POLARIS (https://hkunlp.github.io/blog/2025/Polaris)
    """

    def __init__(self, writer: BufferWriter, variance_threshold: float = 0.0, **kwargs) -> None:
        super().__init__(writer)
        self.variance_threshold = variance_threshold

    async def add(self, experiences: List[Experience], step: int) -> Tuple[int, Dict]:
        cnt = 0
        metrics = {}
        cnt_tot = len(experiences)
        effective_tasks, effective_experiences = [], []
        with Timer(metrics, "add_strategy_time"):
            grouped_experiences = group_by(experiences, id_type="task")
            for task_id, group_exps in grouped_experiences.items():
                if len(group_exps) < 2:
                    continue
                # check if the rewards are the same
                rewards = [exp.reward for exp in group_exps]
                variance = np.var(rewards)
                if variance <= self.variance_threshold:
                    continue
                cnt += len(group_exps)
                effective_tasks.append(task_id)
                effective_experiences.extend(group_exps)

            if not effective_tasks:
                return 0, metrics

        task_ids_to_add = effective_tasks.copy()
        task_id_offset = len(grouped_experiences)
        while cnt < cnt_tot:
            if not task_ids_to_add:
                task_ids_to_add = effective_tasks.copy()
                random.shuffle(task_ids_to_add)
                task_id_offset += len(grouped_experiences)
            task_id = task_ids_to_add.pop()

            copied_exps = copy.deepcopy(grouped_experiences[task_id])

            for exp in copied_exps:
                exp.eid.task += task_id_offset

            cnt += len(copied_exps)
            effective_experiences.extend(copied_exps)

        await self.writer.write_async(effective_experiences)
        return cnt, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"variance_threshold": 0.0}
