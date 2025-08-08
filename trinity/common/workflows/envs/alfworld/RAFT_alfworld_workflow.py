# -*- coding: utf-8 -*-
import re
from datetime import datetime
from typing import Dict, List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow
from trinity.utils.log import get_logger

logger = get_logger(__name__)
logger.setLevel("INFO")

ALFWORLD_SYSTEM_PROMPT = """
You are an agent interacting with a virtual text-based environment.

## Response Format:
You MUST use this exact format for every response. All three tags are REQUIRED in sequential order:

<experience>Working principles, strategies, common knowledge, and potential pitfalls relevant to the current task.</experience>\n\n
<think>your reasoning process</think>\n\n
<action>exactly one action command</action>

## Action Commands:
  look:                             look around your current location
  inventory:                        check your current inventory(you can only have 1 item in your inventory)
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):    place an object in or on a receptacle
  examine (something):              examine a receptacle or an object
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object

For example your output should be like this:
<experience>In household tasks, I should start by exploring the environment to understand available objects and receptacles. Common pitfalls include forgetting to check inventory capacity and not examining objects before taking action.</experience>\n\n<think> To solve the task, I need first to ... </think>\n\n<action>go to cabinet 1</action>

## Important Note:
You must ensure that the <experience> section contains descriptions of working principles, strategies, common sense knowledge, and potential pitfalls that are universally applicable to this type of task, rather than generic statements, placeholder content, or overly specific behavioral guidelines.
"""


def format_observation(observation: str):
    if "Nothing happens." in observation:
        observation += "Please check if the action you take is valid or you have carefully followed the action format."
    return "Observation: " + observation


def parse_response(response):
    """Parse all three components from response with a single regex"""
    try:
        # Use single regex to extract all three components at once
        pattern = r"<experience>\s*(.*?)\s*</experience>.*?<think>\s*(.*?)\s*</think>.*?<action>\s*(.*?)\s*</action>"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return {
                "experience": match.group(1).strip(),
                "think": match.group(2).strip(),
                "action": match.group(3).strip(),
            }
        else:
            return {"experience": "", "think": "", "action": ""}
    except Exception as e:
        logger.warning(f"Error parsing response: {e}")
        return {"experience": "", "think": "", "action": ""}


@WORKFLOWS.register_module("RAFT_alfworld_workflow")
class RAFTAlfworldWorkflow(Workflow):
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

        print(
            f"Initializing RAFTAlfworldWorkflow with RAFT learning, temperature={self.temperature}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.game_file_path = task.task_desc or task.raw_task.get("game_file", "")
        self.is_eval = task.is_eval

    def validate_response_format(self, parsed: Dict[str, str]) -> bool:
        """Validate if parsed response has valid content in all required fields"""
        has_think = len(parsed["think"].strip()) > 0
        has_experience = len(parsed["experience"].strip()) > 0
        has_action = len(parsed["action"].strip()) > 0

        return has_think and has_experience and has_action

    def create_environment(self, game_file):
        """Create alfworld environment"""
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import (
                AlfredDemangler,
                AlfredExpert,
                AlfredExpertType,
            )

            expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
            request_infos = textworld.EnvInfos(
                description=True, inventory=True, admissible_commands=True
            )

            env_id = textworld.gym.register_game(
                game_file, request_infos, wrappers=[AlfredDemangler(), expert]
            )
            env = textworld.gym.make(env_id)
            return env

        except Exception as e:
            error_message = f"Error importing AlfworldTWEnv {str(e)}. Please make sure you have installed the alfworld package successfully, following the instructions in https://github.com/alfworld/alfworld"
            raise ImportError(error_message)

    def run_single_rollout(
        self, env
    ) -> tuple[List[Dict[str, str]], float, bool, int, List[Dict[str, str]]]:
        """Run a single rollout with RAFT-guided actions"""
        observation, info = env.reset()
        trajectory = []
        parsed_steps = []  # Store parsed experience, think, action for each step
        action_history = []  # Track last 3 actions for repetition detection

        trajectory.append({"role": "system", "content": ALFWORLD_SYSTEM_PROMPT})

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
            experience_text = parsed["experience"]
            think_text = parsed["think"]
            action_text = parsed["action"]

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

    def generate_default_empty_experience(
        self, msg: str = "Unknown error", info=None, metrics=None
    ) -> Experience:
        """Generate a default empty experience when errors occur"""
        if info is None:
            info = {"error_reason": msg, "rollout_failed": True}
        if metrics is None:
            metrics = {"success": 0.0, "reward": 0.0}

        return Experience(
            tokens=torch.tensor([0], dtype=torch.long),
            prompt_length=0,
            action_mask=torch.tensor([False], dtype=torch.bool),
            info=info,
            metrics=metrics,
        )

    def eval_alfworld(self) -> List[Experience]:
        """Evaluate a single alfworld trajectory"""
        env = self.create_environment(self.game_file_path)
        try:
            trajectory, reward, done, steps, parsed_steps = self.run_single_rollout(env)
        except Exception as e:
            logger.warning(f"Single rollout failed during eval: {e}")
            env.close()
            return [self.generate_default_empty_experience(f"Eval rollout failed: {str(e)}")]
        env.close()

        # Save eval data
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        success = done and reward >= 1

        # Convert trajectory to experience
        experience = self.generate_default_empty_experience(
            msg="Eval completed successfully",
            info={"task_id": task_id, "success": success, "reward": reward, "steps": steps},
            metrics={"success": float(success), "reward": float(reward), "steps": float(steps)},
        )

        return [experience]

    def run(self) -> List[Experience]:
        """Run the RAFT alfworld workflow and return experiences"""

        if self.is_eval:
            return self.eval_alfworld()

        env = self.create_environment(self.game_file_path)

        # Single rollout execution with RAFT guidance
        try:
            trajectory, reward, done, steps, parsed_steps = self.run_single_rollout(env)
        except Exception as e:
            logger.warning(f"Single rollout failed: {e}")
            env.close()
            return [self.generate_default_empty_experience(f"Training rollout failed: {str(e)}")]
        env.close()

        # Determine success based on Alfworld's reward system
        success = done and reward >= 1
        traj_format_valid = True
        for step in parsed_steps:
            if not self.validate_response_format(step):
                traj_format_valid = False
                break

        print(f"Task result: done={done}, reward={reward:.3f}, steps={steps}, success={success}")

        if reward >= 1 and traj_format_valid:
            print("✅ Task completed successfully in the first attempt!")
            try:
                experience = self.process_messages_to_experience(
                    trajectory, info={"success": success, "reward": reward, "steps": steps}
                )
            except Exception as e:
                logger.warning(f"Failed to convert trajectory to experience: {e}")
                # Create default zero experience
                experience = self.generate_default_empty_experience(
                    f"Experience conversion failed: {str(e)}"
                )
            return [experience]
        elif not traj_format_valid and reward >= 1:
            print(
                "❌ Task completed but trajectory format is invalid, skipping SFT data generation."
            )
        else:
            print("❌ Task failed.")

        experience = self.generate_default_empty_experience(
            "Experience conversion failed: Trajectory format invalid",
            metrics={"success": float(success), "reward": float(reward), "steps": float(steps)},
        )
        return [experience]

    def process_messages_to_experience(self, messages, info=None) -> Experience:
        """Convert messages to experience for training"""
        if info is None:
            info = {}

        converted_experience = self.model.convert_messages_to_experience(messages)

        metrics = {}
        for k, v in info.items():
            if isinstance(v, float) or isinstance(v, int):
                metrics[k] = float(v)
        converted_experience.info = info
        converted_experience.metrics = metrics

        return converted_experience

    def resettable(self) -> bool:
        """Indicate that this workflow can be reset to avoid re-initialization"""
        return True

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
