"""Monitor"""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import wandb

from trinity.utils.log import get_logger


class Monitor:
    """Monitor"""

    def __init__(
        self,
        project: str,
        name: str,
        role: str,
        config: Any = None,
    ) -> None:
        self.logger = wandb.init(
            project=project,
            group=name,
            name=f"{name}_{role}",
            tags=[role],
            config=config,
            save_code=False,
        )
        self.console_logger = get_logger(__name__)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        experiences_table = wandb.Table(dataframe=experiences_table)
        self.log(data={table_name: experiences_table}, step=step)

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

    def log(self, data: dict, step: int) -> None:
        """Log metrics."""
        self.logger.log(data, step=step)
        self.console_logger.info(f"Step {step}: {data}")

    def __del__(self) -> None:
        self.logger.finish()
