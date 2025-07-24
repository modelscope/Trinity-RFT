"""Monitor"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import wandb
from torch.utils.tensorboard import SummaryWriter

from trinity.common.config import Config
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

MONITOR = Registry("monitor")


def gather_metrics(metric_list: List[Dict], prefix: str) -> Dict:
    df = pd.DataFrame(metric_list)
    numeric_df = df.select_dtypes(include=[np.number])
    stats_df = numeric_df.agg(["mean", "max", "min"])
    metric = {}
    for col in stats_df.columns:
        metric[f"{prefix}/{col}/mean"] = stats_df.loc["mean", col]
        metric[f"{prefix}/{col}/max"] = stats_df.loc["max", col]
        metric[f"{prefix}/{col}/min"] = stats_df.loc["min", col]
    return metric


class Monitor(ABC):
    """Monitor"""

    def __init__(
        self,
        project: str,
        name: str,
        role: str,
        config: Config = None,  # pass the global Config for recording
    ) -> None:
        self.project = project
        self.name = name
        self.role = role
        self.config = config

    @abstractmethod
    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        """Log a table"""

    @abstractmethod
    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""

    @abstractmethod
    def close(self) -> None:
        """Close the monitor"""

    def __del__(self) -> None:
        self.close()

    def calculate_metrics(
        self, data: dict[str, Union[List[float], float]], prefix: Optional[str] = None
    ) -> dict[str, float]:
        metrics = {}
        for key, val in data.items():
            if prefix is not None:
                key = f"{prefix}/{key}"

            if isinstance(val, List):
                if len(val) > 1:
                    metrics[f"{key}/mean"] = np.mean(val)
                    metrics[f"{key}/max"] = np.amax(val)
                    metrics[f"{key}/min"] = np.amin(val)
                elif len(val) == 1:
                    metrics[key] = val[0]
            else:
                metrics[key] = val
        return metrics


@MONITOR.register_module("tensorboard")
class TensorboardMonitor(Monitor):
    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        self.tensorboard_dir = os.path.join(config.monitor.cache_dir, "tensorboard", role)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.logger = SummaryWriter(self.tensorboard_dir)
        self.console_logger = get_logger(__name__)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        pass

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        for key in data:
            self.logger.add_scalar(key, data[key], step)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        self.logger.close()


@MONITOR.register_module("wandb")
class WandbMonitor(Monitor):
    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        if not group:
            group = name
        self.logger = wandb.init(
            project=project,
            group=group,
            name=f"{name}_{role}",
            tags=[role],
            config=config,
            save_code=False,
        )
        self.console_logger = get_logger(__name__)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        experiences_table = wandb.Table(dataframe=experiences_table)
        self.log(data={table_name: experiences_table}, step=step)

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        self.logger.log(data, step=step, commit=commit)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        self.logger.finish()
