# -*- coding: utf-8 -*-
"""AlgorithmManager for switching between SFT and RFT."""

from trinity.algorithm.algorithm import ALGORITHM_TYPE
from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import ENTROPY_LOSS_FN
from trinity.algorithm.kl_fn.kl_fn import KL_FN
from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN
from trinity.common.config import AlgorithmConfig, Config


class AlgorithmManager:
    def __init__(self, config: Config):
        self.config = config
        sft_type = ALGORITHM_TYPE.get("sft")
        sft_default_config = sft_type.default_config()
        self.sft_algorithm_config = AlgorithmConfig(
            algorithm_type="sft",
            **sft_default_config,
        )
        policy_fn_cls = POLICY_LOSS_FN.get(self.sft_algorithm_config.policy_loss_fn)
        self.sft_algorithm_config.policy_loss_fn_args = policy_fn_cls.default_args()
        kl_loss_fn_cls = KL_FN.get(self.sft_algorithm_config.kl_loss_fn)
        self.sft_algorithm_config.kl_loss_fn_args = kl_loss_fn_cls.default_args()
        entropy_loss_fn_cls = ENTROPY_LOSS_FN.get(self.sft_algorithm_config.entropy_loss_fn)
        self.sft_algorithm_config.entropy_loss_fn_args = entropy_loss_fn_cls.default_args()

    def get_current_algorithm_config(self, global_steps: int):
        if global_steps <= self.config.buffer.trainer_input.sft_warmup_steps:
            return self.sft_algorithm_config
        else:
            return self.config.algorithm

    def need_save(self, global_steps: int):
        return global_steps == self.config.buffer.trainer_input.sft_warmup_steps
