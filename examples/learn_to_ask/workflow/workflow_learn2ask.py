# -*- coding: utf-8 -*-
""" the learn2ask Workflow"""

from __future__ import annotations

import re
import time
from typing import List, Optional

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, SimpleWorkflow, Task
from trinity.utils.log import get_logger

logger = get_logger(__name__)

"""
For ablation studies, you may set the `taskset.workflow_args.train_mode` to:
- Ra+Rs: the default setting,
- Ra: without Rs,
- Rs: without Ra.

Also, you can choose the reward `taskset.workflow_args.fusion_mode` to:
- default: using the multiplicative fusion function,
- sum: using the sum fusion function.
"""


@WORKFLOWS.register_module("learn2ask_workflow")
class Learn2AskWorkflow(SimpleWorkflow):
    """A workflow for Elem training with local model."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self.train_mode = task.workflow_args.get("train_mode", "Ra+Rs")
        self.fusion_mode = task.workflow_args.get("fusion_mode", "default")
        assert (
            auxiliary_models is not None and len(auxiliary_models) == 1
        ), "Please provide one `auxiliary_models` in explorer config for `learn2ask_workflow`."
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    @property
    def resettable(self):
        return True

    def reset(self, task: Task):
        if self.train_mode == "Ra":  # we have a different system prompt for this training mode.
            from trinity.plugins.prompt_learn2ask import (
                rollout_prompt_med_Ra as system_prompt,
            )
        else:  # other modes use the same system prompt
            from trinity.plugins.prompt_learn2ask import (
                rollout_prompt_med as system_prompt,
            )

        self.format_args = task.format_args
        self.system_prompt = system_prompt
        self.reply_prefix = task.format_args.reply_prefix

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.action_truth = (
            task.raw_task["decision_truth"] if "decision_truth" in task.raw_task else "continue"  # type: ignore
        )
        self.info_truth = task.raw_task["info_truth"] if "info_truth" in task.raw_task else "None"  # type: ignore

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def format_messages(self):
        """Format messages for the instruct model."""
        if isinstance(self.task_desc, list):
            messages = [{"role": "system", "content": self.system_prompt}] + self.task_desc
        elif isinstance(self.task_desc, str):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.task_desc},
            ]
        else:
            raise ValueError("`task_desc` must be a list or a string")
        return messages

    def parse_tag_string(self, text):
        pattern = r"<(\w+)>(.*?)</\1>"
        matches = re.findall(pattern, text)
        result = {}
        for tag, value in matches:
            result[tag] = value
        return result

    def merge_msg_list(self, msg_list):
        result_str = ""
        for msg in msg_list:
            if msg["role"] == "user":
                result_str += f"patient: {msg['content']}\n"
            if msg["role"] == "assistant":
                result_str += f"doctor: {msg['content']}\n"
        return result_str

    def run(self) -> List[Experience]:
        # TODO: Optimize the generate function
        messages = self.format_messages()

        logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)
        for index, response in enumerate(responses):
            reward = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
            )
            response.reward = reward
            response_text = response.response_text
            res_text = response_text.replace("\n", " ")
            logger.info(
                f"cid: {self.raw_task.get('cid', 'xxx')}, repeat: {index}, reward: {response.reward}, response: {res_text}"
            )
        return responses

    def llm_reward(self, response):
        from trinity.plugins.prompt_learn2ask import reward_prompt_med as reward_prompt

        history = self.merge_msg_list(self.task_desc + [{"role": "assistant", "content": response}])
        messages = [
            {"role": "system", "content": reward_prompt.format(self.info_truth)},
            {"role": "user", "content": history},
        ]

        try_count, max_retries = 0, 5
        while try_count <= max_retries:
            try:
                reward_model_stream = False
                client = self.auxiliary_models[0]
                completion = client.chat.completions.create(
                    model=client.model_path, messages=messages, stream=reward_model_stream
                )

                if not reward_model_stream:
                    content = completion.choices[0].message.content
                else:
                    content = ""
                    for chunk in completion:
                        if chunk.choices:
                            content += chunk.choices[0].delta.content
                score_dict = self.parse_tag_string(content)
                return score_dict
            except Exception as e:
                try_count += 1
                if try_count > max_retries:
                    logger.warning("retried too many times, abort task.")
                    return {}
                else:
                    logger.warning(f"error: {e}, response:{response}, retries: {try_count}")
                time.sleep(try_count * 1)

    def reward_fn(self, response):
        """
        content_score: R_a, the reward for response quality
        action_score: R_s, the reward for decision correctness
        format_score: P, the reward for response format
        """

        action_response = "stop" if "<stop />" in response else "continue"
        if self.action_truth == action_response:
            action_score = 1.0
            if self.action_truth == "continue":
                score_dict = self.llm_reward(response=response)
                if score_dict != {}:
                    format_score = float(score_dict.get("format_score", 0.0))
                    content_score = float(score_dict.get("content_score", 0.0))
                else:
                    format_score, content_score = 0.0, 0.0
            else:
                content_score = 1.0
                format_score = 1.0 if response == "<stop />" else 0.0
        else:
            action_score, format_score, content_score = 0.0, 0.0, 0.0

        if self.train_mode == "Ra+Rs":  # the default setting
            final_reward = (
                action_score * (1 + 2 * content_score) + format_score
                if self.fusion_mode != "sum"
                else action_score + content_score + format_score
            )
        elif self.train_mode == "Ra":  # for Ra only (without Rs)
            final_reward = 2 * content_score + format_score
        else:  # for Rs only (without Ra)
            final_reward = action_score * 3 + format_score

        return final_reward
