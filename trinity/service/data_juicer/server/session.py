import os
from functools import partial
from typing import Dict, Tuple

from datasets import Dataset
from jsonargparse import Namespace

from trinity.service.data_juicer.server.utils import (
    DJConfig,
    compute_priority_scores,
    group_scores,
    parse_config,
)
from trinity.utils.log import get_logger


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
        self.config = config
        self.dj_config: Namespace = parse_config(config)
        self.priority_weights = self.config.priority_weights or {
            "difficulty": -0.7,
            "diversity": 0.8,
            "usage_frequency": -0.5,
            "quality": 1.0,
        }
        self.order_method = self.config.order_method
        self.order_args = self.config.order_args or {
            "folding_layers": 3,
        }

        self.logger = get_logger(__name__)

    def process_experience(self, ds: Dataset) -> Tuple[Dataset, Dict]:
        """Process a batch of experiences.

        Args:
            ds (Dataset): The input dataset containing a batch of experiences.

        Returns:
            Tuple[Dataset, Dict]: The processed dataset and extracted metrics.
        """
        from data_juicer.core.data import NestedDataset
        from data_juicer.core.executor.default_executor import DefaultExecutor

        dj_executor = DefaultExecutor(cfg=self.dj_config)

        ds = dj_executor.run(NestedDataset(ds))
        metrics = extract_metrics(ds)
        return ds, metrics

    def process_task(self) -> Dict:
        """
        Process task datasets using Data-Juicer
        """
        from data_juicer.core.executor.default_executor import DefaultExecutor

        dj_executor = DefaultExecutor(cfg=self.dj_config)

        ds: Dataset = dj_executor.run()
        # compute priority
        ds = group_scores(ds)
        compute_priority_scores_func = partial(
            compute_priority_scores, priority_weights=self.priority_weights
        )
        ds = ds.map(compute_priority_scores_func)
        # sort the output dataset in priority
        ds = self.order_task(ds)
        # export to the target directory
        ds.to_json(os.path.join(self.config.output_dir, "output.jsonl"))  # type: ignore [arg-type]
        return {"sample_num": ds.num_rows}

    def order_task(self, dataset: Dataset) -> Dataset:
        """
        Order the dataset with specified method.
        """
        # check if priority field exists
        if "priority" not in dataset.features and self.order_method in {"sort", "folding"}:
            self.logger.warning(
                f'"priority" field not found for {self.order_method}. Use "keep" instead.'
            )
            self.order_method = "keep"

        # get top-k
        top_k = self.config.top_k
        if top_k == -1:
            top_k = dataset.num_rows

        if self.order_method == "keep":
            # keep the original order
            return dataset
        elif self.order_method == "shuffle":
            # shuffle the dataset
            return dataset.shuffle()
        elif self.order_method == "sort":
            # sort the dataset acording to priority
            return dataset.sort("priority", reverse=True).take(top_k)
        elif self.order_method == "folding":
            # folding the dataset to repeat the curriculum learning
            # Reference: https://arxiv.org/abs/2506.21545
            sorted_dataset = dataset.sort("priority", reverse=True).take(top_k)
            folding_layers = self.order_args.get("folding_layers", 3)
            folding_indices = []
            for j in range(folding_layers):
                partition = list(range(j, dataset.num_rows, folding_layers))
                folding_indices.extend(partition)
            return sorted_dataset.select(folding_indices)
        else:
            raise ValueError(f"Invalid order method: {self.order_method}")
