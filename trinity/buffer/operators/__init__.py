from trinity.buffer.operators.experience_operator import ExperienceOperator
from trinity.utils.registry import Registry

EXPERIENCE_OPERATORS: Registry = Registry(
    "experience_operators",
    default_mapping={
        "reward_filter": "trinity.buffer.operators.filters.reward_filter.RewardFilter",
        "reward_std_filter": "trinity.buffer.operators.filters.reward_filter.RewardSTDFilter",
        "reward_shaping_mapper": "trinity.buffer.operators.mappers.reward_shaping_mapper.RewardShapingMapper",
        "pass_rate_calculator": "trinity.buffer.operators.mappers.pass_rate_calculator.PassRateCalculator",
        "data_juicer": "trinity.buffer.operators.data_juicer_operator.DataJuicerOperator",
    },
)

__all__ = [
    "ExperienceOperator",
    "EXPERIENCE_OPERATORS",
]
