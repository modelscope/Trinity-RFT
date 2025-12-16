# -*- coding: utf-8 -*-
"""
This file defines a multi-step workflow for the FrozenLake environment.
Modified from https://github.com/rllm-org/rllm/blob/main/rllm/environments/frozenlake/frozenlake.py
"""

from __future__ import annotations

from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, Workflow


class FrozenLakeWorkflow(Workflow):
    """
    FrozenLake environment for multi-step workflows.

    ## Description
    The game starts with the player at random location of the frozen lake grid world with the
    goal located at another random location for the 4x4 environment.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.
    NOTE the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
    use action_map to map from custom action to action defined in FrozenLakeEnv in gymnasium
    - 0: Still
    - 1: Left
    - 2: Down
    - 3: Right
    - 4: Up

    ## Starting State
    The episode starts with the player at random location

    ## Rewards
    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Arguments
    `is_slippery`: if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3

    ## Example
    P   _   _   _
    _   _   _   O
    O   _   O   _
    O   _   _   G
    """

    can_reset: bool = False  # GymFrozenLakeEnv can only reset the player position, not the environment configuration.
    is_async: bool = True
    can_repeat: bool = False

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        """Initialize the FrozenLake workflow.

        Args:
            model: The model wrapper to use for generating actions.
            task: The task configuration containing workflow-specific arguments.
            auxiliary_models: Optional list of auxiliary models.
        """
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )

        # Extract workflow-specific arguments
        workflow_args = task.workflow_args if hasattr(task, "workflow_args") else {}
        self.env_max_steps = workflow_args.get("env_max_steps", 8)
        self.agent_max_steps = workflow_args.get("agent_max_steps", 10)
        self.desc = workflow_args.get("desc", None)
        self.is_slippery = workflow_args.get("is_slippery", False)
        self.max_response_tokens = self.rollout_args.get("max_response_tokens", 10240)

        # Extract task-specific arguments
        self.raw_task = task.raw_task if hasattr(task, "raw_task") else {}
        self.size = self.raw_task.get("size", 1)
        self.p = self.raw_task.get("p", 0.8)
        self.seed = self.raw_task.get("seed", 42)

        from agentscope.model import OpenAIChatModel

        from examples.agentscope_frozenlake.agent import FrozenLakeAgent
        from examples.agentscope_frozenlake.env import FrozenLakeEnv

        self.agentscope_model = OpenAIChatModel(
            api_key="EMPTY",
            model_name=model.model_path,
            generate_kwargs=self.rollout_args,
            stream=False,
        )

        self.agentscope_model.client = self.model.get_openai_async_client()
        self.agent = FrozenLakeAgent(model=self.agentscope_model, max_steps=self.agent_max_steps)
        self.env = FrozenLakeEnv(
            max_steps=self.env_max_steps,
            desc=self.desc,
            is_slippery=self.is_slippery,
            size=self.size,
            p=self.p,
            seed=self.seed,
        )

    @property
    def rollout_args(self):
        return {
            "temperature": self.task.rollout_args.temperature,
            "max_tokens": self.task.rollout_args.max_tokens,
        }

    async def run_async(self) -> List[Experience]:
        self.env.reset(self.task.raw_task)
        terminate_reason = None
        observation_str = str(self.env.render())
        rewards = []
        step_count = 0
        done = False
        for _ in range(self.agent_max_steps):
            step_count += 1
            try:
                action = await self.agent.step(current_observation=observation_str)
            except Exception as e:
                self.logger.error(f"Agent failed to produce action due to error: {e}")
                terminate_reason = "agent_error"
                break
            observation, reward, done, _ = self.env.step(action)
            observation_str = str(observation)
            rewards.append(reward)
            if done:
                terminate_reason = "success"
                break

        if terminate_reason is None:
            terminate_reason = "max_steps_reached"

        final_reward = sum(rewards)
        exps = self.model.extract_experience_from_history()
        for exp in exps:
            exp.reward = final_reward
            exp.info["terminate_reason"] = terminate_reason

        if len(exps) > 0:
            exps[-1].metrics = {
                "env_steps": step_count,
                "env_done": int(done),
                "final_reward": final_reward,
            }
        return exps
