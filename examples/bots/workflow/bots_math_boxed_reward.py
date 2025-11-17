from typing import Optional

from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS, RewardFn
from trinity.utils.eval_utils import validate_think_pattern


@REWARD_FUNCTIONS.register_module("bots_math_boxed_reward")
class BOTSMathBoxedRewardFn(RewardFn):
    """A reward function that rewards for math task for BOTS."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        pass

    def __call__(  # type: ignore
        self,
        response: str,
        truth: Optional[str] = None,
        with_think: Optional[bool] = False,
        format_score_coef: Optional[float] = 0.1,
        **kwargs,
    ) -> dict[str, float]:
        from trinity.plugins.bots_reward import compute_score

        accuracy_score = compute_score(response, truth)

        format_score = 0.0
        if with_think and not validate_think_pattern(response):
            format_score = (format_score_coef or 0.1) * -1.0

        return {"accuracy": accuracy_score, "format_score": format_score}
