# -*- coding: utf-8 -*-
"""We include the DAPO math workflows in this file."""

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

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        rollout_args = asdict(task.rollout_args)
        self.rollout_args = rollout_args
        self.is_eval = task.is_eval

        self.workflow_args = task.workflow_args

        self.reward_fn = MathDAPORewardFn(
            enable_overlong_penalty=self.workflow_args.get("enable_overlong_penalty", None),
            penalty_factor=self.workflow_args.get("penalty_factor", None),
            max_response_length=self.workflow_args.get("max_response_length", None),
            cache_length=self.workflow_args.get("cache_length", None),
        )

    def run(self) -> List[Experience]:
        messages = self.format_messages()

        logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)

        for response in responses:
            reward = self.reward_fn(  # type: ignore # TODO: fix type
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
                return_dict=self.is_eval,
                response_token=response.tokens[response.prompt_length :],
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

    def format_messages(self):
        messages = [{"role": "user", "content": self.task_desc}]
        return messages
