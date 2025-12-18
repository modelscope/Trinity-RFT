# -*- coding: utf-8 -*-
"""Math Reward Function Class."""
from typing import Optional

from trinity.common.rewards.accuracy_reward import AccuracyReward
from trinity.common.rewards.eval_utils import (
    compute_score_v0,
    simple_answer_parser,
    validate_think_pattern,
)
from trinity.common.rewards.format_reward import FormatReward
from trinity.common.rewards.reward_fn import RewardFn


class MathRewardFn(RewardFn):
    """A reward function that rewards for math task."""

    DEFAULT_FORMAT_PATTERN = r".*?<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    DEFAULT_ANSWER_PARSER = simple_answer_parser

    def __init__(
        self,
        answer_parser=DEFAULT_ANSWER_PARSER,
        pattern=DEFAULT_FORMAT_PATTERN,
    ) -> None:
        self.accuracy_reward = AccuracyReward(answer_parser)
        self.format_reward = FormatReward(pattern)

    def __call__(  # type: ignore
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
    ) -> dict[str, float]:
        accuracy_score = self.accuracy_reward(response, prompt, truth)

        format_score = self.format_reward(response)

        return {**accuracy_score, **format_score}


class MathBoxedRewardFn(RewardFn):
    """A reward function that rewards for math task."""

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
        accuracy_score = compute_score_v0(response, truth)

        format_score = 0.0
        if with_think and not validate_think_pattern(response):
            format_score = (format_score_coef or 0.1) * -1.0

        return {"accuracy": accuracy_score, "format_score": format_score}
