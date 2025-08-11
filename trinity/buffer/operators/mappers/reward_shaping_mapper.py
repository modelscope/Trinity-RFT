from typing import Dict, List, Optional, Tuple

from trinity.buffer.operators import EXPERIENCE_OPERATORS, ExperienceOperator
from trinity.common.constants import OpType
from trinity.common.experience import Experience


@EXPERIENCE_OPERATORS.register_module("reward_shaping_mapper")
class RewardShapingMapper(ExperienceOperator):
    """
    Re-shaping the existing rewards of experiences based on rules or other advanced methods.

    Note: This mapper assumes that the reward is already calculated and stored in the Experience object,
        and the necessary stats are already calculated and stored in the Experience info field.
    """

    def __init__(self, reward_shaping_configs: Optional[List[Dict]] = None):
        """
        Initialization method.

        :param reward_shaping_configs: the configs for reward shaping. Must be a list of dict, where the dict should
            include 3 fields:
                - stats_key: the field key name of target stats used to shape the reward.
                - op_type: the type of operator to applied between the reward and the target stats. Should be one of
                    {"ADD", "SUB", "MUL", "DIV"}
                - weight: the weight for the target stats.
            For example:
            [
                {
                    "stats_key": "quality_score",
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
        for exp in exps:
            # skip experiences that don't have reward
            if exp.reward is None:
                continue
            res_exp = exp
            for reward_shaping_config in self.reward_shaping_configs:
                res_exp = self._reward_shaping_single(res_exp, reward_shaping_config)
            res_exps.append(res_exp)
        return res_exps, {}

    def _reward_shaping_single(self, exp: Experience, reward_shaping_config: Dict):
        """
        Re-shaping the existing reward of one experience based on the given reward_shaping_config.
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
