# -*- coding: utf-8 -*-
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.utils.log import get_logger

logger = get_logger(__name__)


# Setup Jinja2 environment for prompt templates
def get_jinja_env():
    """Get Jinja2 environment for loading templates"""
    prompt_dir = os.path.join(os.path.dirname(__file__), "RAFT_prompt")
    return Environment(loader=FileSystemLoader(prompt_dir))


def format_observation(observation: str):
    """Format observation string with additional guidance"""
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


def validate_response_format(parsed: Dict[str, str]) -> bool:
    """Validate if parsed response has valid content in all required fields"""
    has_think = len(parsed["think"].strip()) > 0
    has_experience = len(parsed["experience"].strip()) > 0
    has_action = len(parsed["action"].strip()) > 0

    return has_think and has_experience and has_action


def validate_trajectory_format(parsed_steps: List[Dict[str, str]]) -> bool:
    """Validate if all steps in a trajectory have valid format"""
    for step in parsed_steps:
        if not validate_response_format(step):
            return False
    return True


def generate_default_empty_experience(
    msg: str = "Unknown error", info=None, metrics=None
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


def create_alfworld_environment(game_file):
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


def process_messages_to_experience(model, messages, info=None) -> Experience:
    """Convert messages to experience for training, with fallback to default empty experience"""
    if info is None:
        info = {}

    try:
        converted_experience = model.convert_messages_to_experience(messages)

        metrics = {}
        for k, v in info.items():
            if isinstance(v, float) or isinstance(v, int):
                metrics[k] = float(v)
        converted_experience.info = info
        converted_experience.metrics = metrics

        return converted_experience
    except Exception as e:
        logger.warning(f"Failed to convert messages to experience: {e}")
        return generate_default_empty_experience(
            f"Experience conversion failed: {str(e)}",
            info=info,
            metrics={k: float(v) for k, v in info.items() if isinstance(v, (float, int))},
        )


def save_task_data(
    task_id: str,
    game_file_path: str,
    first_trajectory: List[Dict[str, str]],
    first_reward: float,
    first_steps: int,
    first_success: bool,
    second_trajectory: Optional[List[Dict[str, str]]],
    second_reward: Optional[float],
    second_steps: Optional[int],
    second_success: Optional[bool],
    kept_for_sft: bool,
    is_eval: bool,
    eval_dir: str,
    sft_dir: str,
    non_sft_dir: str,
    training_data: Optional[List[Dict[str, str]]] = None,
):
    """Save detailed exploration data to individual task file in appropriate folder"""
    task_data = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "game_file": game_file_path,
        # First exploration
        "first_exploration": {
            "trajectory": first_trajectory,
            "reward": first_reward,
            "steps": first_steps,
            "success": first_success,
        },
        # Second exploration
        "second_exploration": {
            "trajectory": second_trajectory,
            "reward": second_reward,
            "steps": second_steps,
            "success": second_success,
        },
        # Training data (clean dialogue format for SFT)
        "training_data": training_data if training_data is not None else "",
        "kept_for_sft": kept_for_sft,
    }

    # Determine folder based on eval mode and SFT data status (following webshop pattern)
    if is_eval:
        target_dir = eval_dir
    elif kept_for_sft:
        target_dir = sft_dir
    else:
        target_dir = non_sft_dir

    task_file_path = os.path.join(target_dir, f"{task_id}.json")

    with open(task_file_path, "w", encoding="utf-8") as f:
        json.dump(task_data, f, ensure_ascii=False, indent=2)
