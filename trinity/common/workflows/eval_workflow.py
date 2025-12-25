# -*- coding: utf-8 -*-
"""Evaluation Workflow Class"""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.config import GenerationConfig
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.qwen25_eval import verify_math_answer
from trinity.common.workflows.workflow import Task, Workflow


class MathEvalWorkflow(Workflow):
    """
    A workflow for standard math evaluation.

    The evaluation standard and prompting style are follow the Qwen2.5-Math
    model's evaluation methodology. For more details on their approach, see:
    https://github.com/QwenLM/Qwen2.5-Math
    """

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

        self.raw_task = task.raw_task
        self.truth = task.truth

        # TODO: customize the config in the yaml
        self.eval_gen_args = asdict(GenerationConfig(temperature=0.6, top_p=0.8, logprobs=0, n=1))

    def format_messages(self):
        """Format message for the evaluation of qwen_boxed type."""
        if not self.raw_task or "question" not in self.raw_task:
            raise ValueError("Raw task data must contain a 'question' field for MathEvalWorkflow.")

        problem_input = self.raw_task["question"]

        system_prompt = "You are a helpful assistant."
        user_prompt = f"{problem_input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def run(self) -> List[Experience]:
        messages = self.format_messages()

        responses: List[Experience] = self.model.chat(messages, **self.eval_gen_args)

        for response in responses:
            if response.response_text is None or self.task.truth is None:
                continue

            accuracy, _ = verify_math_answer(
                response_text=response.response_text, ground_truth=self.task.truth
            )

            acc_metrics = {"accuracy": accuracy}
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(acc_metrics)

        return responses


class AsyncMathEvalWorkflow(MathEvalWorkflow):
    is_async: bool = True

    async def run_async(self) -> List[Experience]:
        messages = self.format_messages()

        responses: List[Experience] = await self.model.chat_async(messages, **self.eval_gen_args)

        for response in responses:
            if response.response_text is None or self.task.truth is None:
                continue

            accuracy, _ = verify_math_answer(
                response_text=response.response_text, ground_truth=self.task.truth
            )

            acc_metrics = {"accuracy": accuracy}
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(acc_metrics)

        return responses
