"""Base Reward Function Class."""
import json
from typing import Optional

from trinity.common.rewards.eval_utils import (
    evaluate_equation,
    extract_solution,
    validate_equation,
)
from trinity.common.rewards.reward_fn import RewardFn


class CountDownRewardFn(RewardFn):
    """A reward function that rewards for countdown task.
    Ref: Jiayi-Pan/TinyZero verl/utils/reward_score/countdown.py
    """

    def __init__(self):
        pass

    def __call__(  # type: ignore
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
    ) -> dict[str, float]:
        truth = json.loads(truth)  # type: ignore
        target = truth["target"]  # type: ignore
        numbers = truth["numbers"]  # type: ignore

        solution_str = response
        equation = extract_solution(solution_str=solution_str)
        format_score = 0.1
        score = 1.0

        if equation is None:
            return {"score": 0}

        # Validate equation uses correct numbers
        if not validate_equation(equation, numbers):
            return {"score": format_score}

        # Evaluate equation
        try:
            result = evaluate_equation(equation)
            if result is None:
                return {"score": format_score}

            if abs(result - target) < 1e-5:  # Account for floating point precision
                return {"score": score}
            else:
                return {"score": format_score}
        except Exception as e:  # noqa: F841
            return {"score": format_score}
