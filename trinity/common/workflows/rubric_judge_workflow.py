# -*- coding: utf-8 -*-
"""A workflow with LLM-as-a-judge."""
import json
from typing import List, Optional, Tuple

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, SimpleWorkflow, Task


@WORKFLOWS.register_module("rubric_judge_workflow")
class RubricJudgeWorkflow(SimpleWorkflow):
    """A workflow using LLM-as-a-judge and rubrics to get the reward.

    Adapted from https://arxiv.org/pdf/2507.17746
    """

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    def reset(self, task: Task):
        """Modified from SimpleWorkflow.reset"""
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix

        if self.system_prompt is None:
            self.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    """

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth
        self.rubric = self.raw_task.get("rubric", [])

    def run(self) -> List[Experience]:
        """Modified from SimpleWorkflow.run"""

        messages = self.format_messages()

        self.logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)

        # === RULER scores as rewards ===
        assert (
            self.auxiliary_models is not None
        ), "Current implementation of RULER requires that auxiliary_models is not None."

        judge_success_list = []
        for i, response in enumerate(responses):
            judge_success, reward = self.get_judge_reward(
                response=response.response_text, judger=self.auxiliary_models[0]
            )
            response.reward = reward
            response.eid.run = i + self.run_id_base

            judge_success_list.append(judge_success)

            if i == 0:
                self.logger.debug(
                    f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {response.reward}"
                )

        # record judge success
        judge_success_rate = sum(judge_success_list) / len(judge_success_list)
        for response in responses:
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update({"judge_success": float(judge_success_rate)})

        return responses

    def get_judge_reward(self, response: str, judger: openai.OpenAI) -> Tuple[bool, float]:
        """Get rewards with LLM-as-a-judge
        The prompts are adapted from RAR-IMPLICIT method in https://arxiv.org/pdf/2507.17746
        """
        # Step 1: format prompts
        # system prompt
        ruler_system_prompt = """You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.
Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.
Start your response with a valid JSON object that starts with "```json" and ends with "```". The JSON object should contain
a single key "rating" and the value should be an integer between 1 and 10.
Example response:
```json
{
"rating": 7
}```"""
        # user prompt
        if len(self.rubric) > 0:
            rubric_prompt_parts = [
                f"Rubric {i} (weight: {single_rubric['weight']}): {single_rubric['description']}"
                for i, single_rubric in enumerate(self.rubric)
            ]
            rubric_list_string = "\n".join(rubric_prompt_parts)
        else:
            self.logger.warning("No rubric is provided!")
            rubric_list_string = "Rubrics are not provided."

        ruler_user_prompt = f"""Given the following prompt, response, and rubrics, please rate the overall quality of the response on a scale of 1 to 10 based
on how well it satisfies the rubrics.
<prompt>
{self.task_desc}
</prompt>
<response>
{response}
</response>
<rubrics>
{rubric_list_string}
</rubrics>
Your JSON Evaluation:
""".strip()

        # Step 2: invoke judger LLM
        messages = [
            {"role": "system", "content": ruler_system_prompt},
            {"role": "user", "content": ruler_user_prompt},
        ]
        completion = judger.chat.completions.create(
            model=judger.model_path, messages=messages, stream=False, temperature=0.0
        )
        judger_response = completion.choices[0].message.content
        self.logger.debug(f"LLM judge response: {judger_response}")

        # Step 3: extract score from judger's response (expecting a JSON block with "rating")
        try:
            # Extract content between ```json and ```
            json_start = judger_response.find("```json")
            if json_start == -1:
                json_start = judger_response.find("```")
            json_end = judger_response.find("```", json_start + 3)

            if json_start == -1 or json_end == -1:
                self.logger.warning("No JSON code block found in judger response.")
                return False, 0.0

            json_str = judger_response[json_start:json_end].strip()
            # Remove the opening marker
            if "```json" in json_str:
                json_str = json_str.replace("```json", "", 1)
            elif "```" in json_str:
                json_str = json_str.replace("```", "", 1)
            json_str = json_str.strip()

            # Parse JSON
            parsed = json.loads(json_str)
            rating = parsed.get("rating")

            if not isinstance(rating, (int, float)) or not (1 <= rating <= 10):
                self.logger.warning(f"Invalid or out-of-range rating: {rating}")
                return False, 0.0

            normalized_score = (rating - 1) / 9  # Normalize 1-10 to 0-1 scale
            return True, normalized_score

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from judger response: {e}")
            return False, 0.0
        except Exception as e:
            self.logger.warning(f"Unexpected error when processing judger response: {e}")
            return False, 0.0
