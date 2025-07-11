# -*- coding: utf-8 -*-
"""Base Reward Function Class."""
import json
import re
from typing import Callable, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS, RewardFn
from trinity.utils.eval_utils import (
    compute_score,
    evaluate_equation,
    extract_solution,
    simple_answer_parser,
    validate_equation,
    validate_think_pattern,
)
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@REWARD_FUNCTIONS.register_module("accuracy_reward")
class AccuracyReward(RewardFn):
    """A reward function that rewards correct answers.
    Ref: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
    """

    def __init__(self, answer_parser: Optional[Callable[[str], str]] = None):
        self.answer_parser = answer_parser

    def __call__(  # type: ignore
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
    ) -> dict[str, float]:
        if self.answer_parser:
            answer_parsed = self.answer_parser(response)
            truth_parsed = self.answer_parser(truth)  # type: ignore [arg-type]

        else:
            truth_parsed = parse(
                truth,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(truth_parsed) == 0:
                truth_parsed = truth

            answer_parsed = parse(
                response,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, truth_parsed))
        except Exception as e:
            logger.info(f"verify failed: {e}, answer: {answer_parsed}, gold: {truth_parsed}")
            reward = 0.0
        return {"accuracy": reward}


@REWARD_FUNCTIONS.register_module("format_reward")
class FormatReward(RewardFn):
    """A reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags.
    Ref: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
    """

    def __init__(self, pattern: Optional[str] = None):
        self.pattern = pattern if pattern else r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"

    def __call__(  # type: ignore
        self,
        response,
    ) -> dict[str, float]:
        if re.match(self.pattern, response, re.DOTALL | re.MULTILINE):
            return {"format_score": 0.1}
        else:
            return {"format_score": -0.1}


@REWARD_FUNCTIONS.register_module("math_reward")
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

        return accuracy_score | format_score


@REWARD_FUNCTIONS.register_module("countdown_reward")
class CountDownRewardFn(RewardFn):
    """A reward function that rewards for countdown task."""

    def __init__(self):
        pass

    def __call__(  # type: ignore
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
    ) -> dict[str, float]:
        # Copy from Jiayi-Pan/TinyZero verl/utils/reward_score/countdown.py
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


@REWARD_FUNCTIONS.register_module("math_boxed_reward")
class MathBoxedRewardFn(RewardFn):
    """A reward function that rewards for math task."""

    def __init__(
        self,
    ) -> None:
        pass

    def __call__(  # type: ignore
        self,
        response: str,
        prompt: Optional[str] = None,
        truth: Optional[str] = None,
        with_think: Optional[bool] = False,
        format_score_coef: Optional[float] = 0.1,
    ) -> dict[str, float]:
        accuracy_score = compute_score(response, truth)

        format_score = 0.0
        if with_think and not validate_think_pattern(response):
            format_score = (format_score_coef or 0.1) * -1.0

        return {"accuracy": accuracy_score, "format_score": format_score}
