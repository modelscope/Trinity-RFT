from trinity.algorithm.algorithm import ALGORITHM
from trinity.common.config import AlgorithmConfig, Config


class AlgorithmManager:
    def __init__(self, config: Config):
        self.config = config
        sft_type = ALGORITHM.get("sft")
        sft_default_config = sft_type.get_default_config()
        self.sft_algorithm_config = AlgorithmConfig(
            algorithm_type=sft_type,
            **sft_default_config,
        )

    def get_current_algorithm_config(self, global_steps: int):
        if global_steps <= self.config.buffer.trainer_input.sft_warmup_steps:
            return self.sft_algorithm_config
        else:
            return self.config.algorithm.algorithm_type

    def need_save(self, global_steps: int):
        return global_steps == self.config.buffer.trainer_input.sft_warmup_steps
