"""Base Reward Function Class."""

import re
from typing import Optional

from trinity.common.rewards.reward_fn import RewardFn


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
