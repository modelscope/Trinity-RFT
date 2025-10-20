from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import Tuple

from trinity.buffer.operators.experience_operator import (
    EXPERIENCE_OPERATORS,
    ExperienceOperator,
)
from trinity.buffer.task_scheduler import TASKSET_SCHEDULE_METRIC
from trinity.common.experience import Experience


@EXPERIENCE_OPERATORS.register_module("pass_rate_calculator")
class PassRateCalculator(ExperienceOperator):
    def __init__(self, reward_shaping_configs: Optional[List[Dict]] = None):
        self.reward_shaping_configs = reward_shaping_configs

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        raw_metric = defaultdict(list)
        for exp in exps:
            raw_metric[exp.task_index].append(exp.reward)
        metric = {task_index: np.mean(rewards) for task_index, rewards in raw_metric.items()}
        return exps, {TASKSET_SCHEDULE_METRIC: metric}
