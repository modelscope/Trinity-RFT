# -*- coding: utf-8 -*-
"""Configs for RFT."""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from trinity.common.constants import (
    AlgorithmType,
    MonitorType,
    PromptType,
    StorageType,
    SyncMethod,
)
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class FormatConfig:
    """Configuration for data formatting"""

    prompt_key: str = ""
    response_key: str = ""
    chat_template: str = ""

    # for sample-level task controlling
    reward_fn_key: str = ""
    workflow_key: str = ""
    # for math dataset
    solution_key: str = ""

    # for reward dataset
    reward_key: str = ""

    # for dpo dataset
    chosen_key: str = ""
    rejected_key: str = ""

    # for unpaired preference dataset
    label_key: str = ""


@dataclass
class DataConfig:
    """Data config"""

    data_workflow_url: Optional[str] = None

    dataset_path: str = ""
    train_split: str = "train"
    subset_name: Optional[str] = None
    eval_split: Optional[str] = None  # TODO: check data format
    format_config: FormatConfig = field(default_factory=FormatConfig)

    # data active iterator related
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    dj_config_path: Optional[str] = None  # The path to Data-Juicer config file.
    dj_process_desc: Optional[
        str
    ] = None  # Describe the data processing procedure without requiring users to be aware of the specific DJ parameters
    agent_model_name: Optional[str] = None
    agent_model_config: Optional[Dict[str, Any]] = None
    clean_strategy: str = "iterative"
    min_size_ratio: Optional[float] = None
    min_priority_score: Optional[float] = 0.0
    priority_weights: Optional[Dict[str, float]] = None
    data_dist: Optional[str] = "gaussian"  # one of ["gaussian", "uniform"]

    # dataset database related
    db_url: str = ""
    max_retry_times: int = 3
    max_retry_interval: int = 1

    # downstream loading related
    total_epochs: int = 1
    batch_size: int = 1
    default_workflow_type: str = ""
    default_reward_fn_type: str = ""


@dataclass
class ModelConfig:
    # TODO: add more
    # source model path
    model_path: str = ""
    critic_model_path: str = ""
    max_prompt_tokens: int = 2048
    max_response_tokens: int = 2048
    # The checkpoint directory, contains a latest dir link and multiple checkpoint dirs.
    checkpoint_path: str = ""


@dataclass
class ClusterConfig:
    """Config for the cluster."""

    node_num: int = 1
    gpu_per_node: int = 8


@dataclass
class DatasetConfig:
    """The config for a dataset."""

    name: str
    storage_type: StorageType
    algorithm_type: AlgorithmType = AlgorithmType.PPO
    path: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BufferConfig:
    """Config for experience buffer."""

    db_url: Optional[str] = None  # Is deprecated, please set `buffer.train_dataset.path` instead.
    read_batch_size: int = 32
    max_retry_times: int = 3
    max_retry_interval: int = 1
    tokenizer_path: Optional[str] = None
    pad_token_id: Optional[int] = None

    train_dataset: Optional[DatasetConfig] = None
    sft_warmup_dataset: Optional[DatasetConfig] = None

    # remove in the future
    prompt_type: PromptType = PromptType.MESSAGES
    messages_key: str = "messages"
    prompt_key: str = "prompt"
    response_key: str = "response"


@dataclass
class ExplorerConfig:
    """Config for explorer."""

    # inference engine type, `vllm` or `vllm_async`
    engine_type: str = "vllm"

    # number of inference engines
    engine_num: int = 1

    # number of workflow runners.
    # For sync engine (vllm), it should be equal to `engine_num`.
    # For async engine (vllm_async), it can be larger than `engine_num`, e.g. 16 * `engine_num`
    runner_num: int = 1

    # repeat each task for `repeat_times` times (for GPRO-like algorithms)
    repeat_times: int = 1

    # for rollout tokneize
    chat_template: Optional[str] = None

    # for evaluation
    # TODO: remove trainer.eval_interval
    eval_interval: int = 100

    # for vLLM
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = False
    enforce_eager: bool = True
    dtype: str = "bfloat16"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 42
    logprobs: int = 0  # vLLM return `logprobs + 1` elements
    backend: str = "nccl"
    use_ray: bool = False
    gpu_memory_utilization: float = 0.9
    enable_chunked_prefil: bool = False

    # for workflow runner
    max_pending_requests: int = 5
    max_waiting_steps: int = 1
    max_timeout: int = 900  # wait each task for 15 minutes
    max_retry_times: int = 2  # retry each task for 2 times if it fails or timeout


@dataclass
class TrainerConfig:
    trainer_type: str = "verl"
    trainer_config_path: str = ""
    eval_interval: int = 100
    save_interval: int = 0
    enable_preview: bool = True  # enable rollout preview in wandb
    trainer_config: Any = field(default_factory=dict)

    # train algorithm
    algorithm_type: AlgorithmType = AlgorithmType.PPO
    get_exp_strategy: Optional[str] = None

    # warmup config
    sft_warmup_iteration: int = 0


@dataclass
class MonitorConfig:
    # TODO: add more
    project: str = "trinity"
    name: str = "rft"
    monitor_type: MonitorType = MonitorType.WANDB

    # ! DO NOT SET
    # the root directory for cache and meta files, automatically generated
    cache_root_dir: Optional[str] = None
    # directory path for current job, automatically generated
    job_dir: Optional[str] = None


@dataclass
class SynchronizerConfig:
    """Configs for model weight synchronization"""

    # TODO: rename to "checkpoint", "nccl", "ipc"
    sync_method: SyncMethod = SyncMethod.NCCL
    # sync weights every `sync_interval` iterations
    sync_interval: int = 1
    # `sync_iteration_interval` is deprecated, use `sync_interval` instead
    sync_iteration_interval: Optional[int] = None
    sync_timeout: int = 1200
    # wait for the lastest checkpoint to be ready
    wait_for_checkpoint: bool = False
    master_address: Optional[str] = None
    master_port: Optional[int] = None
    explorer_world_size: Optional[int] = None
    backend: str = "nccl"


@dataclass
class Config:
    """Global Configuration"""

    mode: str = "both"  # `explore`, `train` or `both`
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    explorer: ExplorerConfig = field(default_factory=ExplorerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    synchronizer: SynchronizerConfig = field(default_factory=SynchronizerConfig)

    def save(self, config_path: str) -> None:
        """Save config to file."""
        with open(config_path, "w", encoding="utf-8") as f:
            OmegaConf.save(self, f)

    def _check_buffer(self) -> None:
        if self.trainer.sft_warmup_iteration > 0 and self.buffer.sft_warmup_dataset is None:
            raise ValueError(
                "buffer.sft_warmup_dataset is required when trainer.sft_warmup_iteration > 0"
            )
        if self.buffer.db_url:
            raise ValueError(
                "`buffer.db_url` is deprecated, please set `buffer.train_dataset.path` instead."
            )

        if self.buffer.pad_token_id is None:
            from transformers import AutoTokenizer

            try:
                self.buffer.pad_token_id = AutoTokenizer.from_pretrained(
                    self.model.model_path
                ).pad_token_id
            except Exception:
                logger.warning(f"Failed to get pad token id from model {self.model.model_path}")
                self.buffer.pad_token_id = 0
        self.buffer.tokenizer_path = self.model.model_path

        if self.mode == "both":
            if self.buffer.train_dataset is None:
                self.buffer.train_dataset = DatasetConfig(
                    name="experience_buffer",
                    storage_type=StorageType.QUEUE,
                    algorithm_type=self.trainer.algorithm_type,
                )
                logger.info(f"Auto set `buffer.train_dataset` to {self.buffer.train_dataset}")
        else:  # TODO: to be check
            if self.mode == "train" and self.trainer.algorithm_type == AlgorithmType.DPO:
                if self.buffer.train_dataset is None and self.data.dataset_path.strip():
                    self.buffer.train_dataset = DatasetConfig(
                        name="dpo_train_dataset",
                        storage_type=StorageType.FILE,
                        algorithm_type=self.trainer.algorithm_type,
                    )
                    logger.info(f"Auto set `buffer.train_dataset` to {self.buffer.train_dataset}")
            if self.buffer.train_dataset is None:
                raise ValueError("buffer.train_dataset is required when mode is not 'both'")
            self.buffer.train_dataset.algorithm_type = self.trainer.algorithm_type
        if self.buffer.sft_warmup_dataset is not None:
            self.buffer.sft_warmup_dataset.algorithm_type = AlgorithmType.SFT
        self.buffer.read_batch_size = self.data.batch_size * self.explorer.repeat_times

    def check_and_update(self) -> None:  # noqa: C901
        """Check and update the config."""
        # check mode
        if self.mode not in ["explore", "train", "both"]:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.trainer.algorithm_type == AlgorithmType.DPO and self.mode == "both":
            raise ValueError("DPO does not support `both` mode")

        # check model path
        if not os.path.isabs(self.model.model_path):
            self.model.model_path = os.path.join(os.getcwd(), self.model.model_path)
        if not os.path.isabs(self.model.checkpoint_path):
            self.model.checkpoint_path = os.path.join(os.getcwd(), self.model.checkpoint_path)
        if not self.model.critic_model_path:
            self.model.critic_model_path = self.model.model_path

        # check synchronizer
        if self.synchronizer.sync_iteration_interval is not None:
            logger.warning(
                f"`synchronizer.sync_iteration_interval` is deprecated, please use `synchronizer.sync_interval` instead. "
                f"And `synchronizer.sync_interval` will set to {self.synchronizer.sync_iteration_interval} instead."
            )
            self.synchronizer.sync_interval = self.synchronizer.sync_iteration_interval
        assert self.synchronizer.sync_interval > 0
        self.synchronizer.explorer_world_size = (
            self.explorer.engine_num * self.explorer.tensor_parallel_size
        )
        self.synchronizer.backend = self.explorer.backend
        if (
            self.trainer.algorithm_type == AlgorithmType.DPO
            and self.synchronizer.sync_method != SyncMethod.CHECKPOINT
        ):
            self.synchronizer.sync_method = SyncMethod.CHECKPOINT
            logger.warning(
                "DPO only supports checkpoint synchronization, set `synchronizer.sync_method` to `checkpoint`."
            )
        if self.synchronizer.sync_method == SyncMethod.NCCL and self.mode != "both":
            raise ValueError("`nccl` synchronization is only supported in both mode.")

        # check eval_interval
        if (
            self.trainer.algorithm_type != AlgorithmType.DPO
            and self.trainer.eval_interval % self.synchronizer.sync_interval != 0
        ):
            self.trainer.eval_interval = (
                max(self.trainer.eval_interval // self.synchronizer.sync_interval, 1)
            ) * self.synchronizer.sync_interval
            logger.warning(
                f"`eval_interval` is not a multiple of `sync_interval`; adjusted to the nearest integer={self.trainer.eval_interval}."
            )
        if self.explorer.eval_interval != self.trainer.eval_interval:
            self.explorer.eval_interval = self.trainer.eval_interval
            logger.warning(
                f"`explorer.eval_interval` is not equal to `trainer.eval_interval`; adjusted to the same value={self.trainer.eval_interval}."
            )

        # check save_interval
        if (
            self.trainer.algorithm_type != AlgorithmType.DPO
            and self.synchronizer.sync_method == SyncMethod.CHECKPOINT
        ):
            if self.trainer.save_interval != self.synchronizer.sync_interval:
                logger.warning(
                    f"When `trainer.algorithm_type != DPO` and `synchronizer.sync_method == checkpoint`, "
                    f"`trainer.save_interval` will be set to "
                    f"`synchronizer.sync_interval = {self.synchronizer.sync_interval}`."
                )
            self.trainer.save_interval = self.synchronizer.sync_interval

        # check monitor
        if not self.monitor.cache_root_dir:
            # create a cache dir in <checkpoint_path>/.cache
            self.monitor.cache_root_dir = os.path.join(self.model.checkpoint_path, ".cache")
        # create a job dir in <checkpoint_path>/.cache/<project>/<name>
        self.monitor.job_dir = os.path.join(
            self.monitor.cache_root_dir, self.monitor.project, self.monitor.name
        )
        try:
            os.makedirs(self.monitor.job_dir, exist_ok=True)
        except Exception:
            logger.warning(
                "Failed to create cache dir, please check "
                f"your checkpoint path: {self.model.checkpoint_path}"
            )

        # check buffer
        self._check_buffer()
        # check and update trainer
        if self.mode != "explore":
            if self.trainer.trainer_type == "verl":
                if self.trainer.trainer_config:
                    from trinity.common.verl_config import veRLConfig

                    trainer_config_schema = OmegaConf.structured(veRLConfig)
                    trainer_config = OmegaConf.merge(
                        trainer_config_schema, self.trainer.trainer_config
                    )
                    self.trainer.trainer_config = OmegaConf.to_object(trainer_config)
                else:
                    if os.path.isfile(self.trainer.trainer_config_path):
                        from trinity.common.verl_config import load_config

                        self.trainer.trainer_config = load_config(self.trainer.trainer_config_path)
                    else:
                        raise ValueError(
                            f"Invalid trainer config path: {self.trainer.trainer_config_path}"
                        )
            else:
                raise ValueError(f"Invalid trainer type: {self.trainer_type}")
            self.trainer.trainer_config.synchronize_config(self)
        else:
            self.trainer.trainer_config = None


def load_config(config_path: str) -> Config:
    """Load the configuration from the given path."""
    # TODO: add test
    schema = OmegaConf.structured(Config)
    yaml_config = OmegaConf.load(config_path)
    try:
        config = OmegaConf.merge(schema, yaml_config)
        return OmegaConf.to_object(config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
