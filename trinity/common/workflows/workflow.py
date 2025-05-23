# -*- coding: utf-8 -*-
"""Base Workflow Class"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Type, Union

import openai
import torch

from trinity.common.config import FormatConfig, GenerationConfig
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.reward_fn import MathRewardFn, RewardFn
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

logger = get_logger(__name__)


WORKFLOWS = Registry("workflows")


@dataclass
class Task:
    """A Task class that defines a task and its associated reward function / workflow."""

    workflow: Type[Workflow]
    format_args: FormatConfig
    rollout_args: GenerationConfig = field(default_factory=GenerationConfig)
    is_eval: bool = False
    reward_fn: Optional[Type[RewardFn]] = None
    raw_task: Optional[dict] = None  # The raw data sample

    def to_workflow(
        self, model: Any, auxiliary_models: Optional[List[openai.OpenAI]] = None
    ) -> Workflow:
        """Convert the task to a workflow.

        Args:
            model (ModelWrapper): The rollout model for the workflow.

        Returns:
            Workflow: The generated workflow object.
        """
        return self.workflow(
            model=model,
            task=self,
            auxiliary_models=auxiliary_models,
        )

    # Deprecated property, will be removed in the future
    @property
    def task_desc(self) -> Union[str, None]:
        prompt_key = self.format_args.prompt_key
        return self.raw_task[prompt_key] if prompt_key in self.raw_task else None  # type: ignore

    # Deprecated property, will be removed in the future
    @property
    def truth(self) -> Union[str, None]:
        response_key = self.format_args.response_key
        return self.raw_task[response_key] if response_key in self.raw_task else None  # type: ignore


class Workflow(ABC):
    """The base workflow class.

    A workflow is a runnable object which generates a list of experiences.
    """

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self.model = model
        self.auxiliary_models = auxiliary_models

    @property
    def resettable(self):
        return False

    def reset(self, task: Task):
        """Reset the workflow."""
        raise NotImplementedError

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
        task: Task,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
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
        task: Task,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self.reset(task)
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )

    @property
    def resettable(self):
        return True

    def reset(self, task: Task):
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        reward_fn = task.reward_fn
        if isinstance(reward_fn, type) and issubclass(reward_fn, RewardFn):
            self.reward_fn: RewardFn = reward_fn()
        else:
            raise ValueError("`reward_fn` must be a subclass of `RewardFn`")
        # Rollout args
        rollout_args = asdict(task.rollout_args)
        self.rollout_args = rollout_args
        self.is_eval = task.is_eval

    def format_messages(self):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.task_desc})
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        return messages

    def run(self) -> List[Experience]:
        # TODO: Optimize the generate function
        messages = self.format_messages()

        logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)
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
        task: Task,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self.reset(task)
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )

    def reset(self, task: Task):
        if task.reward_fn is None:
            task.reward_fn = MathRewardFn
        if task.reward_fn == MathRewardFn and task.format_args.system_prompt is None:
            task.format_args.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.
"""
        # call the SimpleWorkflow.reset
        super().reset(task)
