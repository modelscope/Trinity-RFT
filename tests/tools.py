import os
import unittest
from collections import defaultdict
from typing import Dict, List

import ray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trinity.common.config import (
    Config,
    FormatConfig,
    GenerationConfig,
    StorageConfig,
    load_config,
)
from trinity.common.constants import PromptType


def get_template_config() -> Config:
    config_path = os.path.join(os.path.dirname(__file__), "template", "config.yaml")
    return load_config(config_path)


def get_model_path() -> str:
    path = os.environ.get("MODEL_PATH")
    if not path:
        raise EnvironmentError(
            "Please set `export MODEL_PATH=<your_model_dir>` before running this test."
        )
    return path


def get_checkpoint_path() -> str:
    path = os.environ.get("CHECKPOINT_PATH")
    if not path:
        raise EnvironmentError(
            "Please set `export CHECKPOINT_PATH=<your_checkpoint_dir>` before running this test."
        )
    return path


def get_unittest_dataset_config(
    dataset_name: str = "countdown", split: str = "train"
) -> StorageConfig:
    """Countdown dataset with 17 samples."""
    if dataset_name == "countdown" or dataset_name == "copy_countdown":
        return StorageConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "countdown"),
            split=split,
            enable_progress_bar=False,
            format=FormatConfig(
                prompt_key="question",
                response_key="answer",
            ),
            rollout_args=GenerationConfig(
                n=1,
                temperature=1.0,
                logprobs=0,
            ),
            default_workflow_type="math_workflow",
            default_reward_fn_type="countdown_reward",
        )
    elif dataset_name in {"eval_short", "eval_long"}:
        return StorageConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", dataset_name),
            split="test",
            format=FormatConfig(
                prompt_key="question",
                response_key="answer",
            ),
            rollout_args=GenerationConfig(
                n=1,
                temperature=1.0,
                logprobs=0,
            ),
            default_workflow_type="math_workflow",
            default_reward_fn_type="math_reward",
        )
    elif dataset_name == "gsm8k":
        return StorageConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "gsm8k"),
            split="train",
            format=FormatConfig(
                prompt_key="question",
                response_key="answer",
            ),
            rollout_args=GenerationConfig(
                n=1,
                temperature=1.0,
                logprobs=0,
            ),
            default_workflow_type="math_workflow",
            default_reward_fn_type="math_reward",
        )
    elif dataset_name == "sft_for_gsm8k":
        return StorageConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "sft_for_gsm8k"),
            split="train",
            format=FormatConfig(
                prompt_type=PromptType.PLAINTEXT,
                prompt_key="prompt",
                response_key="response",
            ),
        )
    elif dataset_name == "dpo":
        return StorageConfig(
            name=dataset_name,
            path=os.path.join(os.path.dirname(__file__), "template", "data", "human_like"),
            split="train",
            format=FormatConfig(
                prompt_type=PromptType.PLAINTEXT,
                prompt_key="prompt",
                chosen_key="chosen",
                rejected_key="rejected",
            ),
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


class TensorBoardParser:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._event_files = self._find_event_files(log_dir)
        self._metrics = self._load_metrics()

    def _find_event_files(self, log_dir: str) -> List[str]:
        event_files = []
        for root, _, files in os.walk(log_dir):
            for f in files:
                if f.startswith("events.out.tfevents."):
                    event_files.append(os.path.join(root, f))
        return event_files

    def _load_metrics(self) -> Dict[str, Dict[int, float]]:
        metrics = defaultdict(dict)

        for event_file in self._event_files:
            ea = EventAccumulator(event_file)
            ea.Reload()
            tags = ea.Tags()["scalars"]
            for tag in tags:
                scalars = ea.Scalars(tag)
                for scalar in scalars:
                    step = scalar.step
                    value = scalar.value
                    if step not in metrics[tag] or value > metrics[tag][step]:
                        metrics[tag][step] = value
        return dict(metrics)

    def metric_exist(self, metric_name: str) -> bool:
        return metric_name in self._metrics

    def metric_min_step(self, metric_name: str) -> int:
        return min(self.metric_steps(metric_name))

    def metric_max_step(self, metric_name: str) -> int:
        return max(self.metric_steps(metric_name))

    def metric_steps(self, metric_name: str) -> List[int]:
        if not self.metric_exist(metric_name):
            raise ValueError(f"Metric '{metric_name}' does not exist.")
        return list(self._metrics[metric_name].keys())

    def metric_values(self, metric_name: str) -> List:
        if not self.metric_exist(metric_name):
            raise ValueError(f"Metric '{metric_name}' does not exist.")
        return list(self._metrics[metric_name].values())

    def metric_list(self, metric_prefix: str) -> List[str]:
        return [name for name in self._metrics if name.startswith(metric_prefix)]


class RayUnittestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True, namespace="trinity_unittest")

    @classmethod
    def tearDownClass(cls):
        ray.shutdown(_exiting_interpreter=True)


class RayUnittestBaseAysnc(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True, namespace="trinity_unittest")

    @classmethod
    def tearDownClass(cls):
        ray.shutdown(_exiting_interpreter=True)
