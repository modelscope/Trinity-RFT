from typing import Dict, List, Optional, Tuple

from trinity.buffer.operators import ExperienceOperator
from trinity.common.constants import OpType
from trinity.common.experience import Experience


class RewardShapingMapper(ExperienceOperator):
    """Re-shaping the existing rewards of experiences based on rules or other advanced methods.

    Note:
        This mapper assumes that the reward is already calculated and stored in the Experience object,
        and the necessary stats are already calculated and stored in the Experience info field.
    """

    def __init__(self, reward_shaping_configs: Optional[List[Dict]] = None):
        """Initializes the RewardShapingMapper.

        Args:
            reward_shaping_configs (list[dict], optional): A list of dictionaries containing reward shaping
                configurations. Each dictionary should include the following keys:

                - stats_key (str): The field key name of target stats used to shape the reward.
                - op_type (str): The type of operator to apply between the reward and the target stats.
                  Should be one of {"ADD", "SUB", "MUL", "DIV"}.
                - weight (float): The weight for the target stats.

                Example:
                    [
                        {
                            "stats_key": "llm_quality_score",
                            "op_type": "ADD",
                            "weight": 1.0,
                        }
                    ]
        """
        if reward_shaping_configs is None:
            reward_shaping_configs = []
        self.reward_shaping_configs = reward_shaping_configs

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        res_exps = []
        reward_diff = []
        for exp in exps:
            # skip experiences that don't have reward
            if exp.reward is None:
                res_exps.append(exp)
                reward_diff.append(0.0)
                continue
            res_exp = exp
            previous_reward = exp.reward
            for reward_shaping_config in self.reward_shaping_configs:
                res_exp = self._reward_shaping_single(res_exp, reward_shaping_config)
            if res_exp.reward is None:
                res_exps.append(exp)
                reward_diff.append(0.0)
                continue
            res_reward = res_exp.reward
            reward_diff.append(res_reward - previous_reward)
            res_exps.append(res_exp)
        if len(reward_diff) == 0:
            return res_exps, {
                "reward_diff/mean": 0,
                "reward_diff/min": 0,
                "reward_diff/max": 0,
            }
        metrics = {
            "reward_diff/mean": 1.0 * sum(reward_diff) / len(reward_diff),
            "reward_diff/min": min(reward_diff),
            "reward_diff/max": max(reward_diff),
        }
        return res_exps, metrics

    def _reward_shaping_single(self, exp: Experience, reward_shaping_config: Dict):
        """Re-shapes the existing reward of one experience based on the given reward_shaping_config.

        Args:
            exp (Experience): The experience object whose reward is to be reshaped.
            reward_shaping_config (dict): A dictionary containing the reward shaping configuration.
                It should include the following keys:
                - stats_key (str): The field key name of target stats used to shape the reward.
                - op_type (str): The type of operator to apply between the reward and the target stats.
                  Should be one of {"ADD", "SUB", "MUL", "DIV"}.
                - weight (float): The weight for the target stats.

        Returns:
            Experience: The experience object with the reshaped reward.
        """
        tgt_stats = reward_shaping_config.get("stats_key", None)
        op_type = OpType[reward_shaping_config.get("op_type", "ADD")]
        weight = reward_shaping_config.get("weight", 1.0)
        # if the target stats is not specified, skip the stats and return the original experience
        if tgt_stats is None:
            return exp
        exp_info = exp.info
        if exp_info is None or len(exp_info) == 0:
            return exp
        # if the target stats does not exist in the exp info, skip the stats and return the original experience
        if tgt_stats not in exp_info:
            return exp
        if op_type == OpType.ADD:
            exp.reward += weight * exp_info[tgt_stats]
        elif op_type == OpType.MUL:
            exp.reward *= weight * exp_info[tgt_stats]
        elif op_type == OpType.SUB:
            exp.reward -= weight * exp_info[tgt_stats]
        elif op_type == OpType.DIV:
            divisor = weight * exp_info[tgt_stats]
            if divisor != 0:
                exp.reward /= divisor
        return exp
