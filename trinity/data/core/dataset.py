from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import networkx as nx
from data_juicer.core.data.dj_dataset import Dataset
from datasets import load_dataset

from trinity.common.config import DataProcessorConfig
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.task import TaskSet
from trinity.common.workflows import WORKFLOWS
from trinity.data.core.formatter import BaseDataFormatter


@dataclass
class RewardSchema:
    """Schema for reward related fields"""

    fields: Dict[str, type]
    weights: Optional[Dict[str, float]] = None


class RftDataset:
    """Dataset class for Reinforcement Fine-Tuning that extends Data-Juicer Dataset.

    Supports:
    1. Multiple reward fields with schema validation
    2. Data lineage tracking
    3. Conversion to different RL training formats
    4. Basic statistics and metrics computation

    Args:
        config (Dict): Configuration dict including DJ config
        reward_schema (Union[str, Dict]): Schema definition for reward fields
        track_lineage (bool): Whether to track data lineage
    """

    def __init__(
        self,
        data_config: DataProcessorConfig,
        reward_schema: Union[str, Dict] = "default",
        track_lineage: bool = True,
    ):
        self.config = data_config
        source_data_path = data_config.source_data_path
        if not source_data_path:
            raise ValueError("source_data_path is not specified in DJ config")
        load_kwargs = data_config.load_kwargs
        self.data = load_dataset(source_data_path, trust_remote_code=True, **load_kwargs)

        self.format = data_config.format

        self.reward_schema = self._init_reward_schema(reward_schema)
        self.stats: Dict[str, Any] = {}

        if track_lineage:
            self.lineage = LineageTracker()

    def format(
        self, formatters: Union[BaseDataFormatter, List[BaseDataFormatter]], num_proc: int = 1
    ):
        if not isinstance(formatters, list):
            formatters = [formatters]
        for formatter in formatters:
            self.data = formatter(self.data, num_proc)

    def to_taskset(self, **kwargs) -> TaskSet:
        default_workflow_cls = WORKFLOWS.get(self.config.default_workflow_type)
        default_reward_fn_cls = REWARD_FUNCTIONS.get(self.config.default_reward_fn_type)
        return TaskSet(
            dataset=self.data,
            config=self.config,
            workflow=default_workflow_cls,
            reward_fn=default_reward_fn_cls,
        )

    def to_parquet(self, path: str):
        self.data.to_parquet(path)

    def to_json(self, path: str):
        self.data.to_json(path, force_ascii=False)

    def _init_reward_schema(self, schema) -> RewardSchema:
        if isinstance(schema, str):
            # Load from predefined schemas
            return self._load_predefined_schema(schema)
        return RewardSchema(**schema)

    def _load_predefined_schema(self, schema_name: str) -> RewardSchema:
        # TODO: Load schema from predefined schemas
        return RewardSchema(fields={}, weights={})

    @property
    def reward_fields(self) -> Dict[str, type]:
        return self.reward_schema.fields

    def merge(self, others: List[Dataset]) -> Dataset:
        """Merge other  datasets"""
        # Implement logic to merge this one to other datasets
        pass

    def compute_metrics(self, metric_names: List[str]) -> Dict[str, float]:
        """TODO: Compute specified metrics for the dataset"""
        results = {}
        for name in metric_names:
            results[name] = self._compute_single_metric(name)
        return results

    def _compute_single_metric(self, metric_name: str) -> float:
        """TODO: Compute a single metric for the dataset"""
        # Implement logic to compute a single metric
        return 0.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # TODO add RFT-related feats (be careful with base classes'propertities, e.g., DJ-Dataset or HF-dataset)
        sample = self.data[idx]
        # self._validate_structure(sample)
        return sample

    def __len__(self):
        return len(self.data)


class LineageTracker(ABC):
    """Base class for tracking data lineage, to be extended by specific implementations"""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_transformation(self, input_data, output_data, transform_info):
        # TODO: Implement this method
        self.graph.add_edge(self._hash(input_data), self._hash(output_data), **transform_info)

    def get_ancestry(self, data):
        return nx.ancestors(self.graph, self._hash(data))
