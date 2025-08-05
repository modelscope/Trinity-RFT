from abc import ABC
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union

import networkx as nx
from datasets import Dataset, concatenate_datasets

from trinity.buffer import get_buffer_reader, get_buffer_writer
from trinity.common.config import BufferConfig, DataPipelineConfig
from trinity.data.core.formatter import BaseDataFormatter
from trinity.utils.log import get_logger

logger = get_logger(__name__)


def dict_to_dataclass(cls, d):
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in valid_keys}
    return cls(**filtered)


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
        data_pipeline_config (DataPipelineConfig): Configuration including DJ config
        reward_schema (Union[str, Dict]): Schema definition for reward fields
        track_lineage (bool): Whether to track data lineage
    """

    def __init__(
        self,
        data_pipeline_config: DataPipelineConfig,
        buffer_config: BufferConfig = None,
        reward_schema: Union[str, Dict] = "default",
        track_lineage: bool = True,
    ):
        self.config = data_pipeline_config
        self.buffer_config = buffer_config
        # init input buffers
        input_buffer_configs = self.config.input_buffers
        if len(input_buffer_configs) == 0:
            raise ValueError("input_buffers is empty in data pipeline config")
        self.input_buffers = []
        for input_buffer_config in input_buffer_configs:
            self.input_buffers.append(get_buffer_reader(input_buffer_config, self.buffer_config))
        # init output buffer
        self.output_buffer = get_buffer_writer(self.config.output_buffer, self.buffer_config)

        self.data = Dataset.from_list([])
        self.original_dataclass = None

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

    def sort_by(self, key: str, reverse: bool = False, top_k: int = -1):
        if top_k == -1:
            top_k = len(self.data)
        self.data = self.data.sort(key, reverse=reverse).take(top_k)

    def read_from_buffer(self):
        datasets = []
        for buffer in self.input_buffers:
            exp_list = buffer.read()
            if len(exp_list) > 0 and is_dataclass(exp_list[0]):
                exp_list = [asdict(exp) for exp in exp_list]
                if self.original_dataclass is None:
                    self.original_dataclass = exp_list[0].__class__
            datasets.append(Dataset.from_list([exp for exp in exp_list]))
        self.data = concatenate_datasets(datasets)
        logger.info(f"Read {len(self.data)} samples from input buffers")

    def write_to_buffer(self):
        if self.original_dataclass is not None:
            exp_list = [dict_to_dataclass(self.original_dataclass, d) for d in self.data.to_list()]
        else:
            exp_list = self.data.to_list()
        self.output_buffer.write(exp_list)
        logger.info(f"Wrote {len(self.data)} samples to output buffer")
        self.data = Dataset.from_list([])

    def release_output_buffer(self):
        self.output_buffer.release()

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
