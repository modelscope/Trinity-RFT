# -*- coding: utf-8 -*-
"""Reward functions for RFT"""

from trinity.common.rewards.dapo_reward import MathDAPORewardFn
from trinity.common.rewards.reward_fn import (
    REWARD_FUNCTIONS,
    AccuracyReward,
    FormatReward,
    RewardFn,
)

__all__ = [
    "RewardFn",
    "REWARD_FUNCTIONS",
    "AccuracyReward",
    "FormatReward",
    "MathDAPORewardFn",
]
