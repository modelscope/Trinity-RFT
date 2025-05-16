# -*- coding: utf-8 -*-
"""Base Workflow Class"""

from abc import ABC, abstractmethod
from typing import List, Optional, Type

import torch

from trinity.common.config import StorageConfig
from trinity.common.constants import TaskType
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.reward_fn import MathRewardFn, RewardFn
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

logger = get_logger(__name__)


WORKFLOWS = Registry("workflows")


class Workflow(ABC):
    """The base workflow class.

    A workflow is a runnable object which generates a list of experiences.
    """

    def __init__(
        self,
        model: ModelWrapper,
        task_desc: str,
        taskset_config: StorageConfig,
        truth: Optional[str] = None,
        reward_fn: Optional[Type[RewardFn]] = None,
        raw_task: Optional[dict] = None,
    ):
        self.model = model

    @abstractmethod
    def run(self) -> List[Experience]:
        """Run workflow and return a list of experiences."""


class MultiTurnWorkflow(Workflow):
    """
    The base workflow class for multi-turn tasks.
    """

    def __init__(
        self,
        model: ModelWrapper,
        task_desc: str,
        taskset_config: StorageConfig,
        truth: Optional[str] = None,
        reward_fn: Optional[Type[RewardFn]] = None,
        raw_task: Optional[dict] = None,
    ):
        super().__init__(
            model,
            task_desc,
            taskset_config,
            truth=truth,
            reward_fn=reward_fn,
            raw_task=raw_task,
        )

    @abstractmethod
    def run(self) -> List[Experience]:
        """Run workflow and return a list of experiences."""

    def process_messages_to_experience(self, messages, reward, info={}) -> Experience:
        converted_experience = self.model.convert_messages_to_experience(messages)

        tokens = converted_experience.tokens
        log_probs = converted_experience.logprobs
        assert converted_experience.action_mask is not None
        generation_mask = converted_experience.action_mask
        log_probs = log_probs * generation_mask

        assert tokens.shape == log_probs.shape
        # set prompt length to the first 1 in the gen_mask
        prompt_length = torch.where(generation_mask == 1)[0][0].item()

        metrics = {}
        for k, v in info.items():
            if isinstance(v, float) or isinstance(v, int):
                metrics[k] = float(v)

        experience = Experience(
            tokens=tokens,
            prompt_length=prompt_length,
            action_mask=generation_mask,
            reward=reward,
            logprobs=log_probs,
            info=info,
            metrics=metrics,
        )
        return experience


@WORKFLOWS.register_module("simple_workflow")
class SimpleWorkflow(Workflow):
    """A workflow for simple single-round task."""

    def __init__(
        self,
        model: ModelWrapper,
        task_desc: str,
        taskset_config: StorageConfig,
        truth: Optional[str] = None,
        reward_fn: Optional[Type[RewardFn]] = None,
        raw_task: Optional[dict] = None,
    ):
        super().__init__(
            model,
            task_desc,
            taskset_config,
            truth=truth,
            reward_fn=reward_fn,
            raw_task=raw_task,
        )
        self.system_prompt = taskset_config.system_prompt
        self.reply_prefix = taskset_config.reply_prefix
        self.task_desc = task_desc
        self.truth = truth
        if isinstance(reward_fn, type) and issubclass(reward_fn, RewardFn):
            self.reward_fn: RewardFn = reward_fn()
        else:
            raise ValueError("`reward_fn` must be a subclass of `RewardFn`")
        # Rollout n times
        self.repeat_times = taskset_config.repeat_times
        self.temperature = taskset_config.temperature
        self.is_eval = taskset_config.task_type == TaskType.EVAL

    def run(self) -> List[Experience]:
        # TODO: Optimize the generate function
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.task_desc})
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})

        logger.debug("start chat")
        responses = self.model.chat(messages, n=self.repeat_times, temperature=self.temperature)
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
        task_desc: str,
        taskset_config: StorageConfig,
        truth: Optional[str] = None,
        reward_fn: Optional[Type[RewardFn]] = None,
        raw_task: Optional[dict] = None,
    ):
        if reward_fn is None:
            reward_fn = MathRewardFn
        if reward_fn == MathRewardFn and taskset_config.system_prompt is None:
            taskset_config.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.
"""
        super().__init__(
            model,
            task_desc,
            taskset_config,
            truth=truth,
            reward_fn=reward_fn,
            raw_task=raw_task,
        )
