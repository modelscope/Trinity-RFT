# -*- coding: utf-8 -*-
"""Base Workflow Class"""

from abc import ABC, abstractmethod
from typing import List

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.reward_fn import MathRewardFn
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

logger = get_logger(__name__)


WORKFLOWS = Registry("workflows")


class Workflow(ABC):
    """The base workflow class.

    A workflow is a runnable object which generates a list of experiences.
    """

    def __init__(self, model: ModelWrapper, **kwargs):
        self.model = model

    @abstractmethod
    def run(self) -> List[Experience]:
        """Run workflow and return a list of experiences."""


@WORKFLOWS.register_module("simple_workflow")
class SimpleWorkflow(Workflow):
    """A workflow for simple single-round task."""

    def __init__(
        self,
        model: ModelWrapper,
        **kwargs,
    ):
        super().__init__(model)
        self.system_prompt = kwargs.get("system_prompt", None)
        self.task_desc = kwargs.get("task_desc")
        self.truth = kwargs.get("truth")
        self.reward_fn = kwargs.get("reward_fn")
        # Rollout n times
        self.repeat_times = kwargs.get("repeat_times", 1)
        self.is_eval = kwargs.get("is_eval", False)

    def run(self) -> List[Experience]:
        # TODO: Optimize the generate function
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.task_desc},
            ]
        else:
            messages = [{"role": "user", "content": self.task_desc}]
        logger.debug("start chat")
        n = 1 if self.is_eval else self.repeat_times
        responses = self.model.chat(messages, n=n)
        for response in responses:
            reward = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
                return_dict=self.is_eval,
            )
            logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
            if isinstance(reward, dict):
                if response.metrics is None:
                    response.metrics = {}
                response.metrics.update(reward)
                reward = sum(reward.values())
            response.reward = reward
        return responses


@WORKFLOWS.register_module("math_workflow")
class MathWorkflow(SimpleWorkflow):
    """A workflow for math tasks as introduced in DeepSeek-R1."""

    def __init__(
        self,
        model: ModelWrapper,
        **kwargs,
    ):
        if kwargs.get("reward_fn", None) is None:
            kwargs["reward_fn"] = MathRewardFn()
            kwargs[
                "system_prompt"
            ] = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.
"""
        super().__init__(
            model,
            **kwargs,
        )
