# -*- coding: utf-8 -*-
"""Task Class."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Type

import datasets
from datasets import Dataset, load_dataset

from trinity.common.config import Config, DataConfig
from trinity.common.constants import TaskType
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.rewards.reward_fn import RewardFn
from trinity.common.schema import RftDatasetModel
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import Workflow
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    """A Task class that defines a task and its associated reward function / workflow."""

    task_desc: str
    workflow: Type[Workflow]
    reward_fn: Optional[RewardFn] = None
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


def task_generator(
    dataset,
    start_index: int,
    config: DataConfig,
    default_workflow: Optional[Type[Workflow]],
    default_reward_fn: Optional[RewardFn],
    task_type: Optional[TaskType],
) -> Iterator[Task]:
    """Get a generator of tasks from the dataset."""
    for i, sample in enumerate(dataset):
        if i < start_index:
            continue

        task_desc = (
            sample[config.format_config.prompt_key]
            if config.format_config.prompt_key in sample
            else None
        )
        truth = (
            sample[config.format_config.response_key]
            if config.format_config.response_key in sample
            else None
        )
        workflow_class = (
            WORKFLOWS.get(sample[config.format_config.workflow_key])
            if config.format_config.workflow_key in sample
            else default_workflow
        )
        reward_fn = (
            REWARD_FUNCTIONS.get(sample[config.format_config.reward_fn_key])
            if config.format_config.reward_fn_key in sample
            else default_reward_fn
        )
        task = Task(
            task_desc=task_desc,
            truth=truth,
            workflow=workflow_class,
            reward_fn=reward_fn,
            raw=sample,
            task_type=task_type,
        )
        yield task


def load_hf_dataset(config: DataConfig, split: str):
    """Load a Hugging Face dataset with optional configuration name."""
    if config.config_name is not None:
        hf_dataset = load_dataset(config.dataset_path, config.config_name, split=split)
    else:
        hf_dataset = load_dataset(config.dataset_path, split=split)
    return hf_dataset


@dataclass
class TaskSet:
    """A TaskSet class that defines a set of tasks and their associated reward functions."""

    dataset: Any  # the source huggingface dataset
    config: DataConfig
    reward_fn: Optional[RewardFn] = None
    workflow: Optional[Type[Workflow]] = None
    task_type: Optional[TaskType] = None
    default_index: int = 0
    default_epoch: int = 0
    total_epoch: int = 1
    _tasks: Iterator[Task] = None
    _index: int = 0
    _epoch: int = 0

    @classmethod
    def load(
        cls, config: DataConfig, latest_task_index: int = 0, task_type: TaskType = None
    ) -> TaskSet:
        """Load the RFT taskset through config."""
        # disable datasets caching to avoid reuse old-version dataset
        datasets.disable_caching()
        if task_type == TaskType.EVAL:
            assert config.eval_split is not None, "eval_split must be provided for eval taskset."
            dataset = load_hf_dataset(config, config.eval_split)
        else:  # default
            if task_type != TaskType.EVAL and config.db_url != "":
                logger.info(f"Loading dataset from database with url: {config.db_url}")
                db_type = config.db_url.split(":")[0]
                db_name = config.db_url.split("/")[-1]
                dataset = Dataset.from_sql(RftDatasetModel.__tablename__, f"{db_type}:///{db_name}")
            elif config.dataset_path != "":
                logger.info(f"Loading dataset from local file with path: {config.dataset_path}.")
                dataset = load_hf_dataset(config, config.train_split)
            else:
                raise ValueError("No dataset path or db url provided.")
        datasets.enable_caching()
        dataset_len = len(dataset)
        default_workflow_cls = WORKFLOWS.get(config.default_workflow_type)
        default_reward_fn_cls = REWARD_FUNCTIONS.get(config.default_reward_fn_type)
        default_reward_instance = default_reward_fn_cls() if default_reward_fn_cls else None
        return cls(
            dataset=dataset,
            config=config,
            workflow=default_workflow_cls,
            reward_fn=default_reward_instance,
            task_type=task_type,
            default_index=latest_task_index % dataset_len,
            default_epoch=latest_task_index // dataset_len,
            total_epoch=config.total_epoch if task_type == TaskType.EXPLORE else 1,
        )

    def __iter__(self) -> Iterator[Task]:
        """Initialize the iterator."""
        self._index = self.default_index
        self._epoch = self.default_epoch
        self._tasks = task_generator(
            self.dataset,
            self.default_index,
            self.config,
            self.workflow,
            self.reward_fn,
            self.task_type,
        )
        return self

    @property
    def index(self) -> int:
        """Get the current index."""
        return self._index

    @property
    def epoch(self) -> int:
        """Get the current epoch."""
        return self._epoch

    def __next__(self) -> Task:
        """Iterate through the tasks in the taskset."""
        if self._epoch >= self.total_epoch:
            raise StopIteration

        try:
            task = next(self._tasks)
            if task.reward_fn is None:
                task.reward_fn = self.reward_fn
            if task.workflow is None:
                task.workflow = self.workflow
            self._index += 1
            return task
        except StopIteration:
            # Reset the task generator and increment the epoch
            self._epoch += 1
            self._index += 1
            if self._epoch >= self.total_epoch:
                raise StopIteration
            self._tasks = task_generator(
                self.dataset,
                0,
                self.config,
                self.workflow,
                self.reward_fn,
                self.task_type,
            )
            return next(self._tasks)
