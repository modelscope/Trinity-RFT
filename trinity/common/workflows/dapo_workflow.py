# -*- coding: utf-8 -*-
"""
We include the DAPO math workflows in this file.
This workflow adopts MathDAPORewardFn as the reward function.
Ref: https://arxiv.org/pdf/2503.14476
"""

from dataclasses import asdict
from typing import List

from trinity.common.experience import Experience
from trinity.common.rewards.dapo_reward import MathDAPORewardFn
from trinity.common.workflows.workflow import WORKFLOWS, SimpleWorkflow, Task
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@WORKFLOWS.register_module("math_dapo_workflow")
class MathDAPOWorkflow(SimpleWorkflow):
    """A workflow for math tasks as introduced in DAPO."""

    def reset(self, task: Task):
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        rollout_args = asdict(task.rollout_args)
        self.rollout_args = rollout_args
        self.is_eval = task.is_eval

        self.workflow_args = task.workflow_args
        self.use_base = self.workflow_args.get("use_base", False)

        self.reward_fn = MathDAPORewardFn(
            enable_overlong_penalty=self.workflow_args.get("enable_overlong_penalty", None),
            penalty_factor=self.workflow_args.get("penalty_factor", None),
            max_response_length=self.workflow_args.get("max_response_length", None),
            cache_length=self.workflow_args.get("cache_length", None),
        )

    def run(self) -> List[Experience]:
        if not self.use_base:
            messages = self.format_messages()
        else:
            prompt_text = self.format_prompt()

        logger.debug("start chat")
        if not self.use_base:
            responses = self.model.chat(messages, **self.rollout_args)
        else:
            responses = self.model.generate([prompt_text], **self.rollout_args)

        for response in responses:
            reward_dict = self.reward_fn(  # type: ignore
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
                response_token=response.tokens[response.prompt_length :],
            )
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            reward = sum(reward_dict.values())
            response.reward = reward

            logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
        return responses
