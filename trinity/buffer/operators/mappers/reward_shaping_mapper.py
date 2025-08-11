from copy import deepcopy
from typing import Optional, List, Tuple, Dict

from trinity.buffer.operators import EXPERIENCE_OPERATORS, ExperienceOperator
from trinity.common.experience import Experience
from trinity.common.config import RewardShapingConfig
from trinity.common.constants import OpType

@EXPERIENCE_OPERATORS.register_module("reward_shaping_mapper")
class RewardShapingMapper(ExperienceOperator):
    """
    Re-shaping the existing rewards of experiences based on rules or other advanced methods.

    Note: This mapper assumes that the reward is already calculated and stored in the Experience object,
        and the necessary stats are already calculated and stored in the Experience info field.
    """

    def __init__(self, reward_shaping_configs: Optional[List[RewardShapingConfig]] = None):
        if reward_shaping_configs is None:
            reward_shaping_configs =[]
        self.reward_shaping_configs = reward_shaping_configs

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        res_exps = []
        for exp in exps:
            # skip experiences that don't have reward
            if exp.reward is None:
                continue
            res_exp = deepcopy(exp)
            for reward_shaping_config in self.reward_shaping_configs:
                res_exp = self._reward_shaping_single(res_exp, reward_shaping_config)
            res_exps.append(res_exp)
        return res_exps, {}

    def _reward_shaping_single(self, exp: Experience, reward_shaping_config: RewardShapingConfig):
        """
        Re-shaping the existing reward of one experience based on the given reward_shaping_config.
        """
        tgt_stats = reward_shaping_config.stats_key
        op_type = reward_shaping_config.op_type
        exp_info = exp.info
        if exp_info is None or len(exp_info) == 0:
            return exp
        # if the target stats does not exist in the exp info, skip the stats and return the original experience
        if tgt_stats not in exp_info:
            return exp
        if op_type == OpType.ADD:
            exp.reward += (reward_shaping_config.weight * exp_info[tgt_stats])
        elif op_type == OpType.MUL:
            exp.reward *= (reward_shaping_config.weight * exp_info[tgt_stats])
        elif op_type == OpType.SUB:
            exp.reward -= (reward_shaping_config.weight * exp_info[tgt_stats])
        elif op_type == OpType.DIV:
            exp.reward /= (reward_shaping_config.weight * exp_info[tgt_stats])
        return exp
