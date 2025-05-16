# -*- coding: utf-8 -*-
"""Task Class."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type

from trinity.common.config import StorageConfig
from trinity.common.rewards.reward_fn import RewardFn
from trinity.common.workflows.workflow import Workflow


@dataclass
class Task:
    """A Task class that defines a task and its associated reward function / workflow."""

    task_desc: str
    workflow: Type[Workflow]
    storage_config: StorageConfig
    reward_fn: Optional[Type[RewardFn]] = None
    truth: Optional[str] = None
    raw: Optional[dict] = None  # The raw data sample

    def to_workflow(self, model: Any) -> Workflow:
        """Convert the task to a workflow.

        Args:
            model (ModelWrapper): The rollout model for the workflow.

        Returns:
            Workflow: The generated workflow object.
        """
        return self.workflow(
            model=model,
            task_desc=self.task_desc,
            truth=self.truth,
            storage_config=self.storage_config,
            reward_fn=self.reward_fn,
            raw=self.raw,
        )
