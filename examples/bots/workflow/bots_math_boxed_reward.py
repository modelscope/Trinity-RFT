from typing import Optional

from examples.bots.workflow.bots_reward import compute_score_bots
from trinity.common.rewards.eval_utils import validate_think_pattern
from trinity.common.rewards.reward_fn import RewardFn


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
        accuracy_score = compute_score_bots(response, truth)

        format_score = 0.0
        if with_think and not validate_think_pattern(response):
            format_score = (format_score_coef or 0.1) * -1.0

        return {"accuracy": accuracy_score, "format_score": format_score}
