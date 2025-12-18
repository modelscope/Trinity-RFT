# -*- coding: utf-8 -*-
"""Base Reward Function Class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from trinity.common.experience import Experience
from trinity.common.rewards.utils import to_rm_gallery_messages


class RewardFn(ABC):
    """Base Reward Function Class."""

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, float]:
        pass


class RMGalleryFn(RewardFn):
    """Reward Function from RMGallery.
    https://github.com/modelscope/RM-Gallery
    """

    def __init__(
        self,
        reward_name,
        **kwargs,
    ):
        from rm_gallery.core.reward.registry import RewardRegistry

        self.reward_model = RewardRegistry.get(reward_name)(**kwargs)

    def __call__(self, experience: Experience, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, float]:  # type: ignore
        """Call the reward function."""

        sample = self._build_sample_from_experience(experience, messages, **kwargs)

        sample_with_reward = self.reward_model.evaluate(sample, **kwargs)

        return self._extract_reward(sample_with_reward)

    def _build_sample_from_experience(
        self, experience: Experience, messages: List[Dict[str, Any]], **kwargs
    ) -> Any:
        """Convert experience to sample.
        Ref: https://github.com/modelscope/RM-Gallery/blob/main/rm_gallery/core/data/schema.py
        """
        from rm_gallery.core.data.schema import DataOutput, DataSample, Step

        output = [
            DataOutput(
                answer=Step(
                    role="assistant",
                    content=str(experience.response_text),
                    label={"reference": kwargs.get("ground_truth", "")},
                ),
            )
        ]

        sample = DataSample(
            unique_id=experience.eid.uid,
            input=to_rm_gallery_messages(messages),
            output=output,
            metadata=experience.info,
        )
        return sample

    def _extract_reward(self, sample: Any) -> Dict[str, float]:
        """
        Extract reward from DataSample in rm-gallery
        """
        reward_dict = {}

        try:
            reward_obj = sample.output[0].answer.reward
        except Exception as e:
            raise ValueError(f"No reward is found in sample: {e}")

        from rm_gallery.core.reward.schema import (
            RewardDimensionWithRank,
            RewardDimensionWithScore,
        )

        if reward_obj.details:
            for detail in reward_obj.details:
                if isinstance(detail, RewardDimensionWithScore):
                    reward_dict[detail.name] = detail.score
                elif isinstance(detail, RewardDimensionWithRank):
                    # TODO: support multi-ranked dimension
                    if detail:
                        top_ranked_item = detail[0]
                        reward_dict[top_ranked_item.name] = top_ranked_item.score
        else:
            reward_dict["reward"] = reward_obj.score

        return reward_dict
