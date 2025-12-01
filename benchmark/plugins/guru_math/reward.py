from typing import Optional

from trinity.common.rewards.math_reward import MathBoxedRewardFn
from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS


@REWARD_FUNCTIONS.register_module("math_boxed_reward_naive_dapo")
class NaiveDapoRewardFn(MathBoxedRewardFn):
    def __call__(  # type: ignore
        self,
        response: str,
        truth: Optional[str] = None,
        with_think: Optional[bool] = False,
        format_score_coef: Optional[float] = 0.1,
        **kwargs,
    ) -> dict[str, float]:
        from .naive_dapo import compute_score

        ret = compute_score(response, truth, None)  # type: ignore
        return {"accuracy": ret["score"], "format_score": 0}


@REWARD_FUNCTIONS.register_module("math_boxed_reward_prime_math")
class PrimeMathRewardFn(MathBoxedRewardFn):
    def __call__(  # type: ignore
        self,
        response: str,
        truth: Optional[str] = None,
        with_think: Optional[bool] = False,
        format_score_coef: Optional[float] = 0.1,
        **kwargs,
    ) -> dict[str, float]:
        from verl.utils.reward_score.prime_math import compute_score

        ret = compute_score(response, truth)
        return {"accuracy": ret["score"], "format_score": 0}
