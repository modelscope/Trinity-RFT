import random
from typing import List

from trinity.common.experience import Experience


def representative_sample(experiences: List[Experience]) -> List[dict]:
    if experiences[0].reward is None:
        sample = random.choice(experiences)
        return [
            {
                "prompt": sample.prompt_text,
                "response": sample.response_text,
            }
        ]
    samples = []
    min_reward_sample = None
    max_reward_sample = None
    for exp in experiences:
        if exp.reward is None:
            continue
        if min_reward_sample is None or exp.reward < min_reward_sample.reward:
            min_reward_sample = exp
        if max_reward_sample is None or exp.reward > max_reward_sample.reward:
            max_reward_sample = exp
    if min_reward_sample is not None:
        samples.append(
            {
                "prompt": min_reward_sample.prompt_text,
                "response": min_reward_sample.response_text,
                "reward": min_reward_sample.reward,
            }
        )
    if max_reward_sample is not None:
        samples.append(
            {
                "prompt": max_reward_sample.prompt_text,
                "response": max_reward_sample.response_text,
                "reward": max_reward_sample.reward,
            }
        )
    return samples
