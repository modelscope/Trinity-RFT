import os

from torch.utils.tensorboard import SummaryWriter

from trinity.common.config import Config
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR, Monitor


@MONITOR.register_module("my_monitor")
class MyMonitor(Monitor):
    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        self.tensorboard_dir = os.path.join(config.monitor.cache_dir, "tensorboard", role)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.logger = SummaryWriter(self.tensorboard_dir)
        self.console_logger = get_logger(__name__)

    def log_table(self, table_name: str, experiences_table, step: int):
        pass

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        for key in data:
            self.logger.add_scalar(key, data[key], step)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        self.logger.close()
