from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from trinity.buffer.operators.experience_operator import ExperienceOperator
from trinity.common.constants import SELECTOR_METRIC
from trinity.common.experience import Experience


class PassRateCalculator(ExperienceOperator):
    def __init__(self, **kwargs):
        pass

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        raw_metric = defaultdict(lambda: defaultdict(list))
        for exp in exps:
            task_index = exp.info["task_index"]
            assert "taskset_id" in task_index
            assert "index" in task_index
            raw_metric[task_index["taskset_id"]][task_index["index"]].append(exp.reward)
        metric = {}
        ref_pass_rates = []
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
            ref_pass_rates.extend(reward_means)
        ret_metric = {SELECTOR_METRIC: metric}

        valid_ratio = np.mean([1 if 0 < pr < 1 else 0 for pr in ref_pass_rates])
        strict_valid_ratio = np.mean(
            [1 if 1 / 16 + 1e-3 < pr < 15 / 16 - 1e-3 else 0 for pr in ref_pass_rates]
        )
        less_than_one_ratio = np.mean([1 if pr < 1 else 0 for pr in ref_pass_rates])
        larger_than_zero_ratio = np.mean([1 if pr > 0 else 0 for pr in ref_pass_rates])
        less_than_15_over_16_ratio = np.mean(
            [1 if pr < 15 / 16 - 1e-3 else 0 for pr in ref_pass_rates]
        )
        larger_than_1_over_16_ratio = np.mean(
            [1 if pr > 1 / 16 + 1e-3 else 0 for pr in ref_pass_rates]
        )
        ret_metric.update(
            {
                "selection/valid_ratio": valid_ratio,
                "selection/strict_valid_ratio": strict_valid_ratio,
                "selection/<1_ratio": less_than_one_ratio,
                "selection/>0_ratio": larger_than_zero_ratio,
                "selection/<15_16_ratio": less_than_15_over_16_ratio,
                "selection/>1_16_ratio": larger_than_1_over_16_ratio,
            }
        )

        return exps, ret_metric
