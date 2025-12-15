# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Dict, List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.alfworld.RAFT_utils import (
    create_alfworld_environment,
    format_observation,
    generate_default_empty_experience,
    get_jinja_env,
    parse_response,
    process_messages_to_experience,
    validate_trajectory_format,
)
from trinity.common.workflows.workflow import Task, Workflow


class RAFTAlfworldWorkflow(Workflow):
    """
    RAFT workflow for alfworld using trajectory context.

    Process:
    1. First exploration with normal experience generation
    2. Generate SFT data from successful attempt
    """

    can_reset: bool = True
    can_repeat: bool = True
    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )
        # Initialize workflow parameters
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.top_k = getattr(task.rollout_args, "top_k", 20)
        self.top_p = getattr(task.rollout_args, "top_p", 0.95)
        self.max_env_steps = 50
        self.max_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval

        # Setup Jinja2 templates
        self.jinja_env = get_jinja_env()
        self.alfworld_system_template = self.jinja_env.get_template("alfworld_system.j2")

        print(
            f"Initializing RAFTAlfworldWorkflow with RAFT learning, temperature={self.temperature}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.game_file_path = task.task_desc or task.raw_task.get("game_file", "")
        self.is_eval = task.is_eval

    def create_environment(self, game_file):
        """Create alfworld environment"""
        return create_alfworld_environment(game_file)

    async def run_single_rollout(
        self, env
    ) -> tuple[List[Dict[str, str]], float, bool, int, List[Dict[str, str]]]:
        """Run a single rollout with RAFT-guided actions"""
        observation, info = env.reset()
        trajectory = []
        parsed_steps = []  # Store parsed experience, think, action for each step
        action_history = []  # Track last 3 actions for repetition detection

        trajectory.append({"role": "system", "content": self.alfworld_system_template.render()})

        # Track the last reward from environment
        last_reward = 0.0

        for step in range(self.max_env_steps):
            trajectory.append({"role": "user", "content": format_observation(observation)})

            # Get model response with RAFT guidance
            responses = await self.model.chat_async(
                trajectory,
                n=1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            response_text = responses[0].response_text.strip()
            trajectory.append({"role": "assistant", "content": response_text})

            # Parse the three components
            parsed = parse_response(response_text)
            experience_text, think_text, action_text = (
                parsed["experience"],
                parsed["think"],
                parsed["action"],
            )

            # Store parsed step for SFT data construction
            parsed_steps.append(
                {
                    "observation": observation,
                    "experience": experience_text,
                    "think": think_text,
                    "action": action_text,
                    "full_response": response_text,
                }
            )

            # Check for consecutive action repetition
            action_history.append(action_text)
            if len(action_history) > 3:
                action_history.pop(0)

            # If last 3 actions are the same, terminate with failure
            if len(action_history) >= 3 and all(
                action == action_history[0] for action in action_history
            ):
                print(f"Terminating due to 3 consecutive identical actions: {action_text}")
                return trajectory, 0.0, False, step + 1, parsed_steps

            # Execute action in environment
            observation, reward, done, info = env.step(action_text)
            last_reward = reward  # Always track the latest reward from environment

            if done:
                return trajectory, reward, done, step + 1, parsed_steps

        # If timeout, return the last reward from environment instead of fixed value
        return trajectory, last_reward, False, self.max_env_steps, parsed_steps

    async def _execute_first_attempt(self) -> tuple:
        """Execute the first attempt and return results"""
        env = self.create_environment(self.game_file_path)

        try:
            trajectory, reward, done, steps, parsed_steps = await self.run_single_rollout(env)
        except Exception as e:
            print(f"Single rollout failed: {e}")
            env.close()
            raise e

        env.close()
        success = done and reward >= 1
        traj_format_valid = validate_trajectory_format(parsed_steps)

        return trajectory, reward, done, steps, parsed_steps, success, traj_format_valid

    async def eval_alfworld(self) -> List[Experience]:
        """Evaluate a single alfworld trajectory"""
        env = self.create_environment(self.game_file_path)
        try:
            trajectory, reward, done, steps, parsed_steps = await self.run_single_rollout(env)
        except Exception as e:
            print(f"Single rollout failed during eval: {e}")
            env.close()
            return [generate_default_empty_experience(f"Eval rollout failed: {str(e)}")]
        env.close()

        # Save eval data
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        success = done and reward >= 1

        # Convert trajectory to experience
        experience = generate_default_empty_experience(
            msg="Eval completed successfully",
            info={"task_id": task_id, "success": success, "reward": reward, "steps": steps},
            metrics={"success": float(success), "reward": float(reward), "steps": float(steps)},
        )

        return [experience]

    async def run_async(self) -> List[Experience]:
        """Run the RAFT alfworld workflow and return experiences"""

        if self.is_eval:
            return await self.eval_alfworld()

        # Execute first attempt
        try:
            (
                trajectory,
                reward,
                done,
                steps,
                parsed_steps,
                success,
                traj_format_valid,
            ) = await self._execute_first_attempt()
        except Exception as e:
            return [generate_default_empty_experience(f"Training rollout failed: {str(e)}")]

        print(f"Task result: done={done}, reward={reward:.3f}, steps={steps}, success={success}")

        if reward >= 1 and traj_format_valid:
            print("✅ Task completed successfully in the first attempt!")
            experience = process_messages_to_experience(
                self.model, trajectory, info={"success": success, "reward": reward, "steps": steps}
            )
            return [experience]
        elif not traj_format_valid and reward >= 1:
            print(
                "❌ Task completed but trajectory format is invalid, skipping SFT data generation."
            )
        else:
            print("❌ Task failed.")

        experience = generate_default_empty_experience(
            "Experience conversion failed: Trajectory format invalid",
            metrics={"success": float(success), "reward": float(reward), "steps": float(steps)},
        )
        return [experience]

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
