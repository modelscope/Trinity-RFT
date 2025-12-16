# -*- coding: utf-8 -*-
"""On-Policy Distillation Workflow.

Reference: Tinker library's on-policy distillation implementation.

Algorithm:
1. Student samples trajectories (with logprobs)
2. Teacher computes logprobs on same trajectories
3. Store teacher_logprobs in experience.info["teacher_logprobs"]
4. Trainer's advantage_fn computes: advantages = teacher_logprobs - student_logprobs
5. Train with importance_sampling loss
"""

from typing import List, Optional

import openai
import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, BaseSimpleWorkflow, Task


@WORKFLOWS.register_module("on_policy_distill_workflow")
class OnPolicyDistillWorkflow(BaseSimpleWorkflow):
    """On-policy distillation workflow.

    Computes and stores teacher_logprobs in experience.info.
    The advantage_fn in trainer will compute:
        advantages = teacher_logprobs - student_logprobs
    """

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
        auxiliary_model_wrappers: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.auxiliary_model_wrappers = auxiliary_model_wrappers

        assert (
            auxiliary_model_wrappers is not None and len(auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)

    async def run_async(self) -> List[Experience]:
        messages = self.format_messages()

        # Step 1: Student samples trajectories
        responses = await self.model.chat_async(messages, **self.rollout_args)

        for i, response in enumerate(responses):
            # Step 2: Teacher computes logprobs
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )

            # Extract response portion
            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs

            # Match lengths
            target_len = len(student_resp_logprobs)
            if len(teacher_resp_logprobs) > target_len:
                teacher_resp_logprobs = teacher_resp_logprobs[:target_len]
            elif len(teacher_resp_logprobs) < target_len:
                padding = torch.zeros(target_len - len(teacher_resp_logprobs))
                teacher_resp_logprobs = torch.cat([teacher_resp_logprobs, padding])

            # Step 3: Store teacher_logprobs for advantage_fn
            response.teacher_logprobs = teacher_resp_logprobs

            # Set a dummy reward (actual advantage computed by advantage_fn)
            response.reward = 0.0
            response.eid.run = i + self.run_id_base

            # Metrics for monitoring
            if response.metrics is None:
                response.metrics = {}
            kl = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            response.metrics["kl_divergence"] = kl

        return responses


@WORKFLOWS.register_module("async_on_policy_distill_workflow")
class AsyncOnPolicyDistillWorkflow(OnPolicyDistillWorkflow):
    pass
