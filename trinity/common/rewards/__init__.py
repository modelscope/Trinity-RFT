# -*- coding: utf-8 -*-
"""Reward functions for RFT"""

# isort: off
from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS, RewardFn, RMGalleryFn

from trinity.common.rewards.accuracy_reward import AccuracyReward
from trinity.common.rewards.countdown_reward import CountDownRewardFn
from trinity.common.rewards.dapo_reward import MathDAPORewardFn
from trinity.common.rewards.format_reward import FormatReward
from trinity.common.rewards.math_reward import MathBoxedRewardFn, MathRewardFn

# isort: on

__all__ = [
    "RewardFn",
    "RMGalleryFn",
    "REWARD_FUNCTIONS",
    "AccuracyReward",
    "CountDownRewardFn",
    "FormatReward",
    "MathRewardFn",
    "MathBoxedRewardFn",
    "MathDAPORewardFn",
]
