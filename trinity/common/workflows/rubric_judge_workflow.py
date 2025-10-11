# -*- coding: utf-8 -*-
"""A workflow for non-verifiable domain with RULER."""
import ast
from typing import List, Optional, Tuple

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.math_reward import MathRewardFn
from trinity.common.workflows.workflow import WORKFLOWS, SimpleWorkflow, Task


@WORKFLOWS.register_module("rubric_judge_workflow")
class MedicineRULERWorkflow(SimpleWorkflow):
    """A workflow for medicine dataset with RULER reward function.

    Modified from `MathRULERWorkflow`.
    Adapted from https://github.com/OpenPipe/ART/blob/main/src/art/rewards/ruler.py
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
        """
        Note that in this workflow, MathRewardFn is a placeholder
        whereas the rewards used by RL training are calculated by RULER.
        """

        if task.reward_fn is None:
            task.reward_fn = MathRewardFn
        if task.reward_fn == MathRewardFn and task.format_args.system_prompt is None:
            task.format_args.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>.
"""
        # call the SimpleWorkflow.reset
        super().reset(task)
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
        judge_success, ruler_scores = self.get_ruler_scores(
            responses=responses, judger=self.auxiliary_models[0]
        )
        for i, response in enumerate(responses):
            response.reward = ruler_scores[i]
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update({"judge_success": float(judge_success)})

            response.eid.run = i + self.run_id_base

            if i == 0:
                self.logger.debug(
                    f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {response.reward}"
                )

        return responses

    def get_ruler_scores(
        self, responses: List[Experience], judger: openai.OpenAI
    ) -> Tuple[bool, List[float]]:
        """Get RULER scores"""

        num_responses = len(responses)

        # Step 1: format prompt for judge
        ruler_system_prompt = f"You are a fair judge. The user will provide a question, {num_responses} candidate solutions, and some rubrics with weights. Your task is to compare the solutions, see how well they resolve the question, and assign a score according to the rubrics within the range [0, 1] for each solution."

        question_prompt = (
            f"Question: {self.task_desc}\n\n"
            f"""Solution format requirement: first thinks about the reasoning process in the mind and then provides the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think>
<answer> answer here </answer>."""
        )

        solutions_prompt_parts = [
            f"Candidate solution {i + 1}: {response.response_text}"
            for i, response in enumerate(responses)
        ]
        solutions_prompt = "\n\n".join(solutions_prompt_parts)

        # Compared with MathRULERWorkflow, we provide rubrics for scoring solutions
        if len(self.rubric) > 0:
            rubric_prompt_parts = [
                f"Rubric {i} (weight: {single_rubric['weight']}): {single_rubric['description']}"
                for i, single_rubric in enumerate(self.rubric)
            ]
            rubric_prompt = "Rubric (checklist and weights):\n" + "\n".join(rubric_prompt_parts)
        else:
            self.logger.warning("No rubric is provided!")
            rubric_prompt = "Rubrics are not provided."

        ruler_user_prompt = f"""
Below is a question, candidate solutions, and rubrics with weights.

{question_prompt}
{solutions_prompt}
{rubric_prompt}

Please assign a score within the range [0, 1] for each of them, reflecting how well they solve the question according to the rubrics.
You may compare them against each other and think step by step before returning your final scores, but keep your reasoning process brief and concise when possible.
Conclude your response with a list of scores, in the following format: [score for solution 1, score for solution 2, ..., score for solution {num_responses}]
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

        # Step 3: extract scores from judger's response
        idx1, idx2 = judger_response.rfind("["), judger_response.rfind("]")
        if (idx1 == -1) or (idx2 == -1) or (idx1 > idx2):
            self.logger.warning(
                "Unable to extract a list from judger response, set scores to all zero."
            )
            return False, [0.0 for _ in range(num_responses)]
        lst_as_str = judger_response[idx1 : (idx2 + 1)]
        try:
            scores = ast.literal_eval(lst_as_str)
            scores = [max(0.0, min(1.0, score)) for score in scores]  # clip to range [0, 1]
            if len(scores) != num_responses:
                self.logger.warning(
                    "The length of list in judger response does not match num_responses."
                )
                return False, [0.0 for _ in range(num_responses)]
            return True, scores
        except Exception:
            self.logger.warning(
                "Unable to parse the list in judger response, set scores to all zero."
            )
            return False, [0.0 for _ in range(num_responses)]
