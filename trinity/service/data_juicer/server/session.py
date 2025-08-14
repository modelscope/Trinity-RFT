from typing import Dict, Tuple

from datasets import Dataset
from jsonargparse import Namespace

from trinity.service.data_juicer.server.utils import DJConfig, parse_config


def extract_metrics(dataset: Dataset) -> Dict:
    """Extract metrics from the processed dataset."""
    return {}


class DataJuicerSession:
    """
    A session for interacting with the Data-Juicer service.
    This class manages the connection and provides methods to send and receive data.
    """

    def __init__(self, config: DJConfig):
        """
        Initialize the DataJuicerSession with a URL and configuration.

        Args:
            config (DataJuicerConfigModel): Configuration parameters provided by Trinity.
        """
        self.config: Namespace = parse_config(config)

    def process(self, ds: Dataset) -> Tuple[Dataset, Dict]:
        # TODO: Implement the processing logic using data juicer executor
        from data_juicer.core.data import NestedDataset
        from data_juicer.core.executor.default_executor import DefaultExecutor

        dj_executor = DefaultExecutor(cfg=self.config)

        ds = dj_executor.run(NestedDataset(ds))
        metrics = extract_metrics(ds)
        return ds, metrics
