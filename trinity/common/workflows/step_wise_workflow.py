from abc import abstractmethod

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow


class StepWiseRewardWorkflow(Workflow):
    """A workflow that implements step-wise rewards for tasks."""

    def __init__(
        self, *, task: Task, model: ModelWrapper, auxiliary_models=None, use_openai_client=True
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        assert model.enable_history, (
            "Rollout Model must have history enabled for step-wise rewards, please "
            "set `explorer.rollout_model.enable_history` to `True` in your config."
        )
        # use the rollout model's OpenAI client to write your agent application
        if use_openai_client:
            self.client: openai.OpenAI = model.get_openai_client()
        else:
            self.client = None

    def run(self) -> list[Experience]:
        """Run the workflow and return a list of experiences with step-wise rewards."""
        experiences = []
        for step in range(self.max_step_num):
            # Run a single step of the agent application
            continue_run = self.step(step_num=step)
            # Collect experiences data of the current step
            exps = self.model.extract_experience_from_history()
            # Calculate the reward for the current step
            reward = self.reward(exps, step_num=step)
            for exp in exps:
                exp.reward = reward
                # set the step number in each experience
                exp.eid.step = step
            # Store the step experiences
            experiences.extend(exps)
            if not continue_run:
                break

        return experiences

    @abstractmethod
    def step(self, step_num: int) -> bool:
        """Run a single step of your agent application.

        Args:
            step_num (int): The current step number.

        Returns:
            bool: Whether to continue running the agent application.

        Tips:
            You can use the openai client (`self.client`) to migrate your existing
            applications at low cost.
        """
        pass

    @abstractmethod
    def reward(self, exps: list[Experience], step_num: int) -> float:
        """Calculate the reward for the given experiences at the specified step."""
        pass

    @property
    @abstractmethod
    def max_step_num(self):
        """Return the maximum number of steps in the task."""

    @property
    def repeatable(self):
        return False


class RewardPropagationWorkflow(Workflow):
    """A workflow that propagates rewards across multiple turns."""

    def __init__(
        self, *, task: Task, model: ModelWrapper, auxiliary_models=None, use_openai_client=True
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        assert model.enable_history, (
            "Rollout Model must have history enabled for step-wise rewards, please "
            "set `explorer.rollout_model.enable_history` to `True` in your config."
        )
        # use the rollout model's OpenAI client to write your agent application
        if use_openai_client:
            self.client: openai.OpenAI = model.get_openai_client()
        else:
            self.client = None

    def run(self) -> list[Experience]:
        """Run the workflow and return a list of experiences with step-wise rewards."""
        experiences = []
        for step in range(self.max_step_num):
            # Run a single step of the agent application
            continue_run = self.step(step_num=step)
            # Collect experiences data of the current step
            exps = self.model.extract_experience_from_history()
            # set the step number in each experience
            for exp in exps:
                exp.eid.step = step
            # Store the step experiences
            experiences.extend(exps)
            if not continue_run:
                break
        reward = self.reward(experiences)
        for exp in experiences:
            exp.reward = reward
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["actual_env_steps"] = step + 1  # +1 because step starts from 0
        return experiences

    @abstractmethod
    def step(self, step_num: int) -> bool:
        """Run a single step of your agent application.

        Args:
            step_num (int): The current step number.

        Returns:
            bool: Whether to continue running the agent application.

        Tips:
            You can use the openai client (`self.client`) to migrate your existing
            applications at low cost.
        """
        pass

    @abstractmethod
    def reward(self, exps: list[Experience]) -> float:
        """Calculate the reward for the given experiences of the entire run."""
        pass

    @property
    @abstractmethod
    def max_step_num(self):
        """Return the maximum number of steps in the task."""

    @property
    def repeatable(self):
        return False
