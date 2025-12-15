# -*- coding: utf-8 -*-
"""Accuracy Reward Function Class."""
from typing import Callable, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig

from trinity.common.rewards.eval_utils import parse_with_timeout, verify_with_timeout
from trinity.common.rewards.reward_fn import RewardFn
from trinity.utils.log import get_logger


class AccuracyReward(RewardFn):
    """A reward function that rewards correct answers.
    Ref: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
    """

    def __init__(self, answer_parser: Optional[Callable[[str], str]] = None):
        self.answer_parser = answer_parser
        self.logger = get_logger(__name__)

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
            truth_parsed = parse_with_timeout(
                pred=truth,  # type: ignore [arg-type]
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(truth_parsed) == 0:
                truth_parsed = truth

            answer_parsed = parse_with_timeout(
                pred=response,
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
            reward = float(verify_with_timeout(answer_parsed, truth_parsed))
        except Exception as e:
            self.logger.info(f"verify failed: {e}, answer: {answer_parsed}, gold: {truth_parsed}")
            reward = 0.0
        return {"accuracy": reward}
