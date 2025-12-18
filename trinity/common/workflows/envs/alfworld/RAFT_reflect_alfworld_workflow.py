# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.alfworld.RAFT_alfworld_workflow import (
    RAFTAlfworldWorkflow,
)
from trinity.common.workflows.envs.alfworld.RAFT_utils import (
    create_alfworld_environment,
    format_observation,
    generate_default_empty_experience,
    generate_reward_feedback,
    parse_response,
    process_messages_to_experience,
    save_task_data,
    validate_trajectory_format,
)
from trinity.common.workflows.workflow import Task


class RAFTReflectAlfworldWorkflow(RAFTAlfworldWorkflow):
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

        # Create data directories specific to reflect workflow
        self.data_dir = "RAFT_reflect_alfworld_data"
        self.sft_dir = os.path.join(self.data_dir, "sft_data")
        self.non_sft_dir = os.path.join(self.data_dir, "non_sft_data")

        os.makedirs(self.sft_dir, exist_ok=True)
        os.makedirs(self.non_sft_dir, exist_ok=True)

        # Setup additional template for second attempt
        self.second_attempt_template = self.jinja_env.get_template("second_attempt_guidance.j2")

        print(
            f"Initializing RAFTReflectAlfworldWorkflow with RAFT learning, temperature={self.temperature}"
        )

    async def construct_sft_data(
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
        ) = await self.re_explore_with_context(first_trajectory, reward, success, original_steps)

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

    async def re_explore_with_context(
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
        reward_feedback = generate_reward_feedback(
            original_reward, original_steps, original_success, self.max_env_steps
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

            responses = await self.model.chat_async(
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

    async def _execute_second_attempt(
        self, trajectory: list, success: bool, reward: float, steps: int
    ) -> Union[tuple[List[Dict[str, str]], Dict[str, Any], List[Dict[str, str]],], Exception]:
        """Execute second attempt and return SFT data"""
        try:
            sft_messages, re_explore_info, new_parsed_steps = await self.construct_sft_data(
                trajectory, success, reward, steps
            )
            return sft_messages, re_explore_info, new_parsed_steps
        except Exception as e:
            print(f"SFT data construction failed: {e}")
            return e

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

    async def run_async(self) -> List[Experience]:
        """Run the RAFT alfworld workflow and return experiences"""

        if self.is_eval:
            return await self.eval_alfworld()

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
            ) = await self._execute_first_attempt()
        except Exception as e:
            return [generate_default_empty_experience(f"Training rollout failed: {str(e)}")]

        # Handle first attempt success cases
        if reward >= 1 and traj_format_valid:
            print("✅ Task completed successfully in the first attempt!")
            experience = process_messages_to_experience(
                self.model, trajectory, info={"success": success, "reward": reward, "steps": steps}
            )
            return [experience]
        elif not traj_format_valid and reward >= 1:
            return self._handle_invalid_format_success(success, reward, steps)

        print(f"Task result: done={done}, reward={reward:.3f}, steps={steps}, success={success}")

        # Execute second attempt
        ret = await self._execute_second_attempt(trajectory, success, reward, steps)
        if isinstance(ret, Exception):
            error = ret
            default_experience = generate_default_empty_experience(
                f"SFT data construction failed: {str(error)}",
            )
            return [default_experience]
        else:
            sft_messages, re_explore_info, new_parsed_steps = ret

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
