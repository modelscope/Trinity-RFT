import os
import unittest
from collections import defaultdict
from typing import Dict, List

import ray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trinity.common.config import Config, DataConfig, FormatConfig, load_config


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


def get_unittest_dataset_config(dataset_name: str = "countdown") -> DataConfig:
    """Countdown sample dataset for 8 iterations"""
    if dataset_name == "countdown":
        return DataConfig(
            total_epochs=2,
            batch_size=4,
            default_workflow_type="math_workflow",
            default_reward_fn_type="countdown_reward",
            dataset_path=os.path.join(os.path.dirname(__file__), "template", "data", "countdown"),
            train_split="train",
            eval_split="test",
            format_config=FormatConfig(
                prompt_key="question",
                response_key="answer",
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

    def metric_max_step(self, metric_name: str) -> int:
        if not self.metric_exist(metric_name):
            raise ValueError(f"Metric '{metric_name}' does not exist.")
        steps = list(self._metrics[metric_name].keys())
        return max(steps)

    def metric_list(self, metric_prefix: str) -> List[str]:
        return [name for name in self._metrics if name.startswith(metric_prefix)]


class RayUnittestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()
