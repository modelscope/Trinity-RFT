# -*- coding: utf-8 -*-
"""Task Class."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type

from trinity.common.config import Config
from trinity.common.constants import TaskType
from trinity.common.rewards.reward_fn import RewardFn
from trinity.common.workflows.workflow import Workflow
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    """A Task class that defines a task and its associated reward function / workflow."""

    task_desc: str
    workflow: Type[Workflow]
    reward_fn: Optional[Type[RewardFn]] = None
    truth: Optional[str] = None
    raw: Optional[dict] = None  # The raw data sample
    task_type: Optional[TaskType] = None

    def to_workflow(self, model: Any, config: Config) -> Workflow:
        """Convert the task to a workflow.

        Args:
            model (ModelWrapper): The rollout model for the workflow.
            config (Config): The global configuration.

        Returns:
            Workflow: The generated workflow object.
        """
        if self.task_type == TaskType.EVAL:
            repeat_times = 1
        else:
            repeat_times = config.explorer.repeat_times
        return self.workflow(
            model=model,
            task_desc=self.task_desc,
            truth=self.truth,
            reward_fn=self.reward_fn,
            raw=self.raw,
            repeat_times=repeat_times,
            config=config,
            is_eval=self.task_type == TaskType.EVAL,
        )
