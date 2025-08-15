import os
from typing import Any, Dict, List, Literal, Optional

from data_juicer.config import get_init_configs, prepare_side_configs
from jsonargparse import Namespace
from pydantic import BaseModel, model_validator


class DJConfig(BaseModel):
    pipeline_type: Literal["task", "experience"] = "experience"

    # For both `task` and `experience`
    operators: Optional[List[Dict[str, Dict[str, Any]]]] = None
    config_path: Optional[str] = None
    np: int = 4

    # For `task` only
    executor_type: Literal["ray", "default"] = "default"
    inputs: List[str] = None  # List of input files
    output_dir: str = None
    output_fields: List[str] = None  # fields in the output dataset

    @model_validator(mode="after")
    def check_dj_config(self):
        if not (self.config_path or self.operators):
            raise ValueError("Must provide at least one of config_path or operators.")
        if self.np <= 0:
            raise ValueError("np must be a positive integer.")
        return self


def parse_config(config: DJConfig) -> Namespace:
    """Convert Trinity config to DJ config"""
    if config.config_path is not None:
        task_config = prepare_side_configs(config.config_path)
        task_config = get_init_configs(task_config)
        return task_config

    if config.pipeline_type == "experience":
        return _parse_experience_pipeline_config(config)
    elif config.pipeline_type == "task":
        return _parse_task_pipeline_config(config)
    else:
        raise ValueError(f"Unknown pipeline type: {config.pipeline_type}")


def _parse_experience_pipeline_config(config: DJConfig) -> Namespace:
    """Parse the experience pipeline configuration."""
    if config.operators is not None:
        exp_config = Namespace(process=[op for op in config.operators], np=config.np)
        exp_config = get_init_configs(exp_config)
    else:
        raise ValueError("At least one of operators or config_path should be provided.")
    return exp_config


def _parse_task_pipeline_config(config: DJConfig) -> Namespace:
    """Parse the task pipeline configuration."""
    if config.operators is not None:
        for input in config.inputs:
            if not os.path.exists(input):
                raise FileNotFoundError(f"{input} does not exist.")
            if not os.path.isfile(input):
                raise ValueError(
                    f"{input} is not a file. Currently, task pipeline only support process on file."
                )
        os.makedirs(config.output_dir, exist_ok=True)
        task_config = Namespace(
            process=[op for op in config.operators],
            np=config.np,
            dataset={
                "configs": [
                    {
                        "type": "local",
                        "weight": 1.0,
                        "path": path,
                    }
                    for path in config.inputs
                ]
            },
            text_keys=config.output_fields,
            export_shard_size=128 * 1024 * 1024,  # 128 MB
        )
        task_config = get_init_configs(task_config)
    else:
        raise ValueError("At least one of operators or config_path should be provided.")
    return task_config
