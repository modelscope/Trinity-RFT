from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from trinity.buffer.operators.experience_operator import (
    EXPERIENCE_OPERATORS,
    ExperienceOperator,
)
from trinity.buffer.task_scheduler import SELECTOR_METRIC
from trinity.common.experience import Experience


@EXPERIENCE_OPERATORS.register_module("pass_rate_calculator")
class PassRateCalculator(ExperienceOperator):
    def __init__(self, reward_shaping_configs: Optional[List[Dict]] = None):
        self.reward_shaping_configs = reward_shaping_configs

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        raw_metric = defaultdict(lambda: defaultdict(list))
        for exp in exps:
            task_index = exp.info["task_index"]
            assert "taskset_id" in task_index
            assert "index" in task_index
            raw_metric[task_index["taskset_id"]][task_index["index"]].append(exp.reward)
        metric = {}
        for taskset_id, taskset_metric in raw_metric.items():
            indices = []
            reward_means = []
            for index, rewards in taskset_metric.items():
                indices.append(index)
                reward_means.append(float(np.mean(rewards)))
            metric[taskset_id] = {
                "indices": indices,
                "values": reward_means,
            }
        return exps, {SELECTOR_METRIC: metric}
