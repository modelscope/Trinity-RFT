# -*- coding: utf-8 -*-
"""
Sudoku Workflow for Trinity-RFT
This workflow demonstrates a simple single-turn task where the model solves a Sudoku puzzle.
"""

from typing import List

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import Task
from trinity.common.workflows.workflow import Workflow


class SudokuWorkflow(Workflow):
    """
    Workflow: SudokuWorkflow
    Purpose: Ask the model to solve a Sudoku puzzle and give reward based on correctness.
    """

    # Workflow does not support reset or repeated runs for now.
    can_reset: bool = False
    can_repeat: bool = False

    def __init__(self, task: Task, model: ModelWrapper, auxiliary_models=None):
        """
        Initialize workflow with:
        - task.raw_task["puzzle"]
        - task.raw_task["solution"]g
        """

        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

        # Extract puzzle input and ground truth
        self.puzzle = task.raw_task.get("puzzle")
        self.solution = task.raw_task.get("solution")

        # Rollout arguments (e.g., temperature, n)
        self.rollout_args = task.rollout_args

    def calculate_reward(self, predicted: str) -> float:
        """
        Reward function:
        Returns 1.0 if predicted output matches solution exactly, else 0.0.
        """
        return 1.0 if predicted.strip() == self.solution.strip() else 0.0

    def run(self) -> List[Experience]:
        """
        Primary execution step of the workflow:
        1. Send puzzle to model
        2. Collect response
        3. Evaluate with reward
        4. Package into Experience list
        """

        responses = self.model.chat(
            [
                {
                    "role": "user",
                    "content": f"Solve this Sudoku puzzle:\n{self.puzzle}",
                }
            ],
            temperature=self.rollout_args.temperature,
        )

        resp = responses[0]  # Single response
        reward = self.calculate_reward(resp.response_text)

        # Return experience in expected format
        return [
            Experience(
                tokens=resp.tokens,
                prompt_length=resp.prompt_length,
                reward=reward,
                logprobs=resp.logprobs,
            )
        ]
