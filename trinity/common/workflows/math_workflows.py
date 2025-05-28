# -*- coding: utf-8 -*-
"""We include seprate the math workflows in this file."""
from typing import List, Optional

import openai

from functools import partial

from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, SimpleWorkflow, BaseModelWorkflow, Task
from trinity.common.rewards.reward_fn import MathRewardFn, MathBoxedRewardFn

PREDEFINED_MATH_SYSTEM_PROMPTS = {
    "deepseek_like": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.""",
    "boxed_with_think": """You are a helpful assistant that solves MATH problems. You should first thinks about the reasoning process in mind and then provides the user with the answer. You should present your reasoning process using the format: <think>\n ...your reasoning process here... </think>\n first. You should always include your final answer in \\boxed{} as closed-form results.""",
    "boxed_no_think": """Please reason step by step, and put your final answer within \\boxed{}.""",
}

@WORKFLOWS.register_module("math_workflow")
class MathWorkflow(SimpleWorkflow):
    """A workflow for math tasks"""

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
        if task.format_args.system_prompt is None:
            task.format_args.system_prompt = PREDEFINED_MATH_SYSTEM_PROMPTS["deepseek_like"]
        if task.format_args.system_prompt in PREDEFINED_MATH_SYSTEM_PROMPTS.keys():
            task.format_args.system_prompt = PREDEFINED_MATH_SYSTEM_PROMPTS[
                task.format_args.system_prompt
            ]
        
        have_boxed_pattern = "boxed{" in task.format_args.system_prompt
        if not have_boxed_pattern:
            task.reward_fn = MathRewardFn
        else:
            have_think_pattern = (
                "</think>" in task.format_args.system_prompt
                and "</think>" in task.format_args.system_prompt
            )
            task.reward_fn = partial(MathBoxedRewardFn, have_think_pattern=have_think_pattern)

        # call the SimpleWorkflow.reset
        super().reset(task)

@WORKFLOWS.register_module("math_based_model_workflow")
class MathBasedModelWorkflow(BaseModelWorkflow):
    """A workflow for math tasks, using base model"""
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
        if task.format_args.system_prompt is None:
            task.format_args.system_prompt = PREDEFINED_MATH_SYSTEM_PROMPTS["deepseek_like"]
        if task.format_args.system_prompt in PREDEFINED_MATH_SYSTEM_PROMPTS.keys():
            task.format_args.system_prompt = PREDEFINED_MATH_SYSTEM_PROMPTS[
                task.format_args.system_prompt
            ]
        
        have_boxed_pattern = "boxed{" in task.format_args.system_prompt
        if not have_boxed_pattern:
            task.reward_fn = MathRewardFn
        else:
            have_think_pattern = (
                "</think>" in task.format_args.system_prompt
                and "</think>" in task.format_args.system_prompt
            )
            task.reward_fn = partial(MathBoxedRewardFn, have_think_pattern=have_think_pattern)

        # call the SimpleWorkflow.reset
        super().reset(task)