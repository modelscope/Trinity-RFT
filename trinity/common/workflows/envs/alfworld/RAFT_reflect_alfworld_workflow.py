# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.alfworld.RAFT_utils import (
    create_alfworld_environment,
    format_observation,
    generate_default_empty_experience,
    get_jinja_env,
    parse_response,
    process_messages_to_experience,
    save_task_data,
    validate_trajectory_format,
)
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow
from trinity.utils.log import get_logger

logger = get_logger(__name__)
logger.setLevel("INFO")


@WORKFLOWS.register_module("RAFT_reflect_alfworld_workflow")
class RAFTReflectAlfworldWorkflow(Workflow):
    """
    RAFT workflow for alfworld using trajectory context.

    Process:
    1. First exploration with normal experience generation
    2. If failed, re-explore with first trajectory as context
    3. Generate SFT data from successful attempt
    """

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

        # Create data directories
        self.data_dir = "RAFT_reflect_alfworld_data"
        self.eval_dir = os.path.join(self.data_dir, "eval")
        self.sft_dir = os.path.join(self.data_dir, "sft_data")
        self.non_sft_dir = os.path.join(self.data_dir, "non_sft_data")

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.sft_dir, exist_ok=True)
        os.makedirs(self.non_sft_dir, exist_ok=True)

        # Setup Jinja2 templates
        self.jinja_env = get_jinja_env()
        self.alfworld_system_template = self.jinja_env.get_template("alfworld_system.j2")
        self.second_attempt_template = self.jinja_env.get_template("second_attempt_guidance.j2")

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

    def run_single_rollout(
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
            responses = self.model.chat(
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

    def construct_sft_data(
        self,
        first_trajectory: List[Dict[str, str]],
        success: bool,
        reward: float,
        original_steps: int,
    ) -> tuple[List[Dict[str, str]], Dict[str, Any], List[Dict[str, str]]]:
        """Generate SFT training data using RAFT learning"""

        # Always perform second attempt with first trajectory as context
        (
            new_trajectory,
            new_reward,
            new_success,
            new_steps,
            new_parsed_steps,
        ) = self.re_explore_with_context(first_trajectory, reward, success, original_steps)

        # Consider improvement if reward is higher OR same reward with fewer steps
        reward_improved = new_reward > reward
        efficiency_improved = new_steps < original_steps

        return (
            new_trajectory,
            {
                "new_reward": new_reward,
                "new_steps": new_steps,
                "reward_improved": reward_improved,
                "efficiency_improved": efficiency_improved,
            },
            new_parsed_steps,
        )

    def generate_reward_feedback(self, reward: float, steps: int, done: bool) -> str:
        """Generate natural language feedback about the attempt's performance"""
        if done and reward >= 1:
            return f"In your attempt, you successfully completed the task in {steps} steps with a reward of {reward:.3f}. Try to maintain this success while being more efficient."
        elif done and reward < 1:
            return f"In your attempt, you completed the task in {steps} steps but only achieved a reward of {reward:.3f}. You need to improve your performance to achieve full success."
        elif not done and steps >= self.max_env_steps:
            return f"In your attempt, you reached the maximum step limit of {self.max_env_steps} steps without completing the task (reward: {reward:.3f}). You need to be more efficient and focused to complete the task within the step limit."
        else:
            return f"In your attempt, you stopped after {steps} steps with a reward of {reward:.3f} without completing the task. You need to improve your strategy and persistence to achieve success."

    def re_explore_with_context(
        self,
        first_trajectory: List[Dict[str, str]],
        original_reward: float,
        original_success: bool,
        original_steps: int,
    ) -> tuple[List[Dict[str, str]], float, bool, int, List[Dict[str, str]]]:
        """Re-explore with first trajectory as context"""

        env = create_alfworld_environment(self.game_file_path)

        observation, info = env.reset()

        # Use first trajectory as context for generation
        context_messages = first_trajectory.copy()

        # Add reward feedback about first attempt
        reward_feedback = self.generate_reward_feedback(
            original_reward, original_steps, original_success
        )
        context_messages.append(
            {
                "role": "system",
                "content": self.second_attempt_template.render(reward_feedback=reward_feedback),
            }
        )

        # Build clean SFT trajectory (like first trajectory format)
        sft_trajectory = [{"role": "system", "content": self.alfworld_system_template.render()}]
        parsed_steps = []  # Track parsed steps for quality analysis

        for step in range(self.max_env_steps):
            # Add to context for generation
            context_messages.append({"role": "user", "content": format_observation(observation)})

            # Add to clean SFT trajectory
            sft_trajectory.append({"role": "user", "content": format_observation(observation)})

            responses = self.model.chat(
                context_messages,
                n=1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            response_text = responses[0].response_text.strip()

            # Parse components for quality analysis
            parsed = parse_response(response_text)
            experience_text, think_text, action_text = (
                parsed["experience"],
                parsed["think"],
                parsed["action"],
            )

            parsed_steps.append(
                {
                    "observation": observation,
                    "experience": experience_text,
                    "think": think_text,
                    "action": action_text,
                    "full_response": response_text,
                }
            )

            # Add to both trajectories
            context_messages.append({"role": "assistant", "content": response_text})
            sft_trajectory.append({"role": "assistant", "content": response_text})

            observation, reward, done, info = env.step(action_text)

            if done:
                env.close()
                return sft_trajectory, reward, done and reward > 0, step + 1, parsed_steps

        env.close()
        return sft_trajectory, reward, False, self.max_env_steps, parsed_steps

    def eval_alfworld(self) -> List[Experience]:
        """Evaluate a single alfworld trajectory"""
        env = create_alfworld_environment(self.game_file_path)
        try:
            trajectory, reward, done, steps, parsed_steps = self.run_single_rollout(env)
        except Exception as e:
            logger.warning(f"Single rollout failed during eval: {e}")
            env.close()
            return [generate_default_empty_experience(f"Eval rollout failed: {str(e)}")]
        env.close()

        # Save eval data
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        success = done and reward >= 1

        save_task_data(
            task_id=task_id,
            game_file_path=self.game_file_path,
            first_trajectory=trajectory,
            first_reward=reward,
            first_steps=steps,
            first_success=success,
            second_trajectory=trajectory,
            second_reward=reward,
            second_steps=steps,
            second_success=success,
            kept_for_sft=False,
            is_eval=self.is_eval,
            eval_dir=self.eval_dir,
            sft_dir=self.sft_dir,
            non_sft_dir=self.non_sft_dir,
            training_data=trajectory,
        )

        # Convert trajectory to experience
        experience = generate_default_empty_experience(
            msg="Eval completed successfully",
            info={"task_id": task_id, "success": success, "reward": reward, "steps": steps},
            metrics={"success": float(success), "reward": float(reward), "steps": float(steps)},
        )

        return [experience]

    def _execute_first_attempt(self, task_id: str) -> tuple:
        """Execute the first attempt and return results"""
        env = create_alfworld_environment(self.game_file_path)

        try:
            trajectory, reward, done, steps, parsed_steps = self.run_single_rollout(env)
        except Exception as e:
            logger.warning(f"Single rollout failed: {e}")
            env.close()
            raise e

        env.close()
        success = done and reward >= 1
        traj_format_valid = validate_trajectory_format(parsed_steps)

        return trajectory, reward, done, steps, parsed_steps, success, traj_format_valid

    def _handle_first_attempt_success(
        self, task_id: str, trajectory: list, reward: float, steps: int, success: bool
    ) -> List[Experience]:
        """Handle successful first attempt"""
        print("✅ Task completed successfully in the first attempt!")

        save_task_data(
            task_id=task_id,
            game_file_path=self.game_file_path,
            first_trajectory=trajectory,
            first_reward=reward,
            first_steps=steps,
            first_success=success,
            second_trajectory=None,
            second_reward=None,
            second_steps=None,
            second_success=None,
            kept_for_sft=False,
            is_eval=self.is_eval,
            eval_dir=self.eval_dir,
            sft_dir=self.sft_dir,
            non_sft_dir=self.non_sft_dir,
            training_data=None,
        )

        experience = process_messages_to_experience(
            self.model, trajectory, info={"success": success, "reward": reward, "steps": steps}
        )
        return [experience]

    def _handle_invalid_format_success(
        self, success: bool, reward: float, steps: int
    ) -> List[Experience]:
        """Handle case where task succeeded but format is invalid"""
        print("❌ Task completed but trajectory format is invalid, skipping SFT data generation.")
        experience = generate_default_empty_experience(
            "Experience conversion failed: Trajectory format invalid",
            metrics={"success": float(success), "reward": float(reward), "steps": float(steps)},
        )
        return [experience]

    def _execute_second_attempt(
        self, trajectory: list, success: bool, reward: float, steps: int
    ) -> tuple:
        """Execute second attempt and return SFT data"""
        try:
            sft_messages, re_explore_info, new_parsed_steps = self.construct_sft_data(
                trajectory, success, reward, steps
            )
            return sft_messages, re_explore_info, new_parsed_steps, None
        except Exception as e:
            logger.warning(f"SFT data construction failed: {e}")
            return None, None, None, e

    def _build_metrics(
        self, reward: float, steps: int, new_parsed_steps: list, re_explore_info: dict
    ) -> dict:
        """Build metrics for tracking"""
        return {
            "reward": float(reward),
            "steps": float(steps),
            "trajectory_length": len(new_parsed_steps),
            "second_reward": float(re_explore_info["new_reward"]),
            "second_steps": float(re_explore_info["new_steps"]),
            "improvement": 1.0 if re_explore_info["reward_improved"] else 0.0,
        }

    def _should_keep_for_sft(self, second_traj_format_valid: bool, re_explore_info: dict) -> bool:
        """Determine if trajectory should be kept for SFT"""
        return second_traj_format_valid and (
            re_explore_info["reward_improved"]
            or (re_explore_info["efficiency_improved"] and re_explore_info["new_reward"] >= 1.0)
        )

    def _generate_experience_from_sft(self, sft_messages: list, metrics: dict) -> Experience:
        """Generate experience from SFT messages"""
        return process_messages_to_experience(self.model, sft_messages, info=metrics)

    def run(self) -> List[Experience]:
        """Run the RAFT alfworld workflow and return experiences"""

        if self.is_eval:
            return self.eval_alfworld()

        # Generate unique task ID using timestamp
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

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
            ) = self._execute_first_attempt(task_id)
        except Exception as e:
            return [generate_default_empty_experience(f"Training rollout failed: {str(e)}")]

        # Handle first attempt success cases
        if reward >= 1 and traj_format_valid:
            return self._handle_first_attempt_success(task_id, trajectory, reward, steps, success)
        elif not traj_format_valid and reward >= 1:
            return self._handle_invalid_format_success(success, reward, steps)

        print(f"Task result: done={done}, reward={reward:.3f}, steps={steps}, success={success}")

        # Execute second attempt
        sft_messages, re_explore_info, new_parsed_steps, error = self._execute_second_attempt(
            trajectory, success, reward, steps
        )
        if error:
            default_experience = generate_default_empty_experience(
                f"SFT data construction failed: {str(error)}",
            )
            return [default_experience]

        # Validate second attempt and build metrics
        second_success = re_explore_info["new_reward"] >= 1
        second_traj_format_valid = validate_trajectory_format(new_parsed_steps)
        metrics = self._build_metrics(reward, steps, new_parsed_steps, re_explore_info)

        # Generate experience if conditions are met
        experiences = []
        kept_for_sft = self._should_keep_for_sft(second_traj_format_valid, re_explore_info)

        if kept_for_sft:
            experience = self._generate_experience_from_sft(sft_messages, metrics)
            experiences.append(experience)
            print(
                f"✅ Generated good training data: orig={reward}, steps={steps}, new={re_explore_info['new_reward']}, new_steps={re_explore_info['new_steps']}"
            )
        else:
            print(
                f"❌ Filtered trajectory: orig={reward}, steps={steps}, new={re_explore_info['new_reward']}, new_steps={re_explore_info['new_steps']}, second_traj_format_valid: {second_traj_format_valid}"
            )

        # Save detailed task data
        save_task_data(
            game_file_path=self.game_file_path,
            is_eval=self.is_eval,
            eval_dir=self.eval_dir,
            sft_dir=self.sft_dir,
            non_sft_dir=self.non_sft_dir,
            task_id=task_id,
            first_trajectory=trajectory,
            first_reward=reward,
            first_steps=steps,
            first_success=success,
            second_trajectory=sft_messages,
            second_reward=re_explore_info["new_reward"],
            second_steps=re_explore_info["new_steps"],
            second_success=second_success,
            kept_for_sft=kept_for_sft,
            training_data=sft_messages,
        )

        # Return default experience if no valid experience generated
        if not experiences:
            experiences.append(generate_default_empty_experience())

        return experiences

    def resettable(self) -> bool:
        """Indicate that this workflow can be reset to avoid re-initialization"""
        return True

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
