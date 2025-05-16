# -*- coding: utf-8 -*-
"""Configs for RFT."""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from trinity.common.constants import (
    AlgorithmType,
    MonitorType,
    PromptType,
    StorageType,
    SyncMethod,
    TaskType,
)
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class FormatConfig:
    """Configuration for data formatting"""

    prompt_type: PromptType = PromptType.MESSAGES

    prompt_key: str = "prompt"
    response_key: str = "response"
    messages_key: str = "message"
    chat_template: str = ""

    # for sample-level task controlling
    reward_fn_key: str = ""
    workflow_key: str = ""
    # for math dataset
    solution_key: str = ""

    # for reward dataset
    reward_key: str = ""

    # for dpo dataset
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"

    # for unpaired preference dataset
    label_key: str = ""


@dataclass
class StorageConfig:
    """Storage config."""

    name: str = ""
    storage_type: StorageType = StorageType.FILE
    algorithm_type: Optional[AlgorithmType] = None  # automatically set
    path: Optional[str] = None

    # used for StorageType.FILE
    split: str = "train"
    subset_name: Optional[str] = None
    format: FormatConfig = field(default_factory=FormatConfig)
    index: int = 0

    # used for algorithm_type is None
    task_type: TaskType = TaskType.EXPLORE
    default_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None
    total_epochs: int = 1  # automatically set
    # used for algorithm_type is None and TaskType.EVAL
    eval_repeat_times: int = 1  # TODO
    eval_temperature: float = 0.1  # TODO


@dataclass
class DataProcessorConfig:
    """Data-Juicer config"""

    data_workflow_url: Optional[str] = None

    source_data_path: str = ""
    format: FormatConfig = field(default_factory=FormatConfig)

    # data active iterator related
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
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


@dataclass
class GlobalConfig:
    # downstream loading related
    total_epochs: int = 1
    batch_size: int = 1
    eval_interval: int = 100
    eval_on_latest_ckp: bool = True


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
    # for models support both thinking and non-thinking mode, e.g., Qwen3
    enable_thinking: bool = False


@dataclass
class ClusterConfig:
    """Config for the cluster."""

    node_num: int = 1
    gpu_per_node: int = 8


@dataclass
class ExplorerInput:
    """Config for explorer input."""

    taskset: StorageConfig = field(default_factory=StorageConfig)
    eval_tasksets: List[StorageConfig] = field(default_factory=list)
    default_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None


@dataclass
class TrainerInput:
    """Config for trainer input."""

    experience_buffer: Optional[StorageConfig] = None
    sft_warmup_dataset: Optional[StorageConfig] = None


@dataclass
class BufferConfig:
    """Config for experience buffer."""

    read_batch_size: int = 32
    max_retry_times: int = 3
    max_retry_interval: int = 1
    tokenizer_path: Optional[str] = None  # automatically set
    pad_token_id: Optional[int] = None  # automatically set

    explorer_input: ExplorerInput = field(default_factory=ExplorerInput)
    explorer_output: Optional[StorageConfig] = None  # currently do not set
    trainer_input: TrainerInput = field(default_factory=TrainerInput)


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
    enable_chunked_prefill: bool = False
    use_v1: bool = True
    bundle_indices: str = ""  # DO NOT SET this field

    # for workflow runner
    max_pending_requests: int = 5
    max_waiting_steps: int = 1
    max_timeout: int = 900  # wait each task for 15 minutes
    max_retry_times: int = 2  # retry each task for 2 times if it fails or timeout


@dataclass
class TrainerConfig:
    trainer_type: str = "verl"
    trainer_config_path: str = ""
    save_interval: int = 0
    enable_preview: bool = True  # enable rollout preview in wandb
    trainer_config: Any = field(default_factory=dict)

    # train algorithm
    algorithm_type: AlgorithmType = AlgorithmType.PPO
    get_exp_strategy: Optional[str] = None

    # warmup config
    sft_warmup_steps: int = 0
    sft_warmup_iteration: Optional[int] = None  # deprecated


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
    # sync weights every `sync_interval` steps
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

    mode: str = "both"  # `explore`, `train`, `both` or `bench`
    data_processor: DataProcessorConfig = field(default_factory=DataProcessorConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
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

    def _check_deprecated(self) -> None:
        if self.synchronizer.sync_iteration_interval is not None:
            logger.warning(
                f"`synchronizer.sync_iteration_interval` is deprecated, please use `synchronizer.sync_interval` instead. "
                f"And `synchronizer.sync_interval` will set to {self.synchronizer.sync_iteration_interval} instead."
            )
            self.synchronizer.sync_interval = self.synchronizer.sync_iteration_interval

        if self.trainer.sft_warmup_iteration is not None:
            logger.warning(
                f"`trainer.sft_warmup_iteration` is deprecated, please use `trainer.sft_warmup_steps` instead. "
                f"And `trainer.sft_warmup_steps` will be set to {self.trainer.sft_warmup_iteration} instead."
            )
            self.trainer.sft_warmup_steps = self.trainer.sft_warmup_iteration

    def _check_interval(self) -> None:
        assert self.synchronizer.sync_interval > 0

        # check eval_interval
        if (
            self.mode != "bench"
            and self.trainer.algorithm_type != AlgorithmType.DPO
            and self.global_config.eval_interval % self.synchronizer.sync_interval != 0
        ):
            self.global_config.eval_interval = (
                max(self.global_config.eval_interval // self.synchronizer.sync_interval, 1)
            ) * self.synchronizer.sync_interval
            logger.warning(
                f"`eval_interval` is not a multiple of `sync_interval`; adjusted to the nearest integer={self.global_config.eval_interval}."
            )

        # check save_interval
        if (
            self.mode != "bench"
            and self.trainer.algorithm_type != AlgorithmType.DPO
            and self.synchronizer.sync_method == SyncMethod.CHECKPOINT
        ):
            if self.trainer.save_interval != self.synchronizer.sync_interval:
                logger.warning(
                    f"When `trainer.algorithm_type` != `DPO` and `synchronizer.sync_method` == `checkpoint`, "
                    f"`trainer.save_interval` will be set to "
                    f"`synchronizer.sync_interval = {self.synchronizer.sync_interval}`."
                )
            self.trainer.save_interval = self.synchronizer.sync_interval

    def _check_buffer(self) -> None:  # noqa: C901
        # check explorer_input
        if self.mode != "train" and self.buffer.explorer_input.taskset.path is None:
            raise ValueError(
                "`buffer.explorer_input.taskset.path` is required, please set it to the path of the taskset."
            )
        self.buffer.explorer_input.taskset.task_type = TaskType.EXPLORE
        self.buffer.explorer_input.taskset.total_epochs = self.global_config.total_epochs
        if self.buffer.explorer_input.taskset.default_workflow_type is None:
            self.buffer.explorer_input.taskset.default_workflow_type = (
                self.buffer.explorer_input.default_workflow_type
            )
        if self.buffer.explorer_input.taskset.default_reward_fn_type is None:
            self.buffer.explorer_input.taskset.default_reward_fn_type = (
                self.buffer.explorer_input.default_reward_fn_type
            )

        for dataset in self.buffer.explorer_input.eval_tasksets:
            dataset.task_type = TaskType.EVAL
            if dataset.default_workflow_type is None:
                dataset.default_workflow_type = self.buffer.explorer_input.default_workflow_type
            if dataset.default_reward_fn_type is None:
                dataset.default_reward_fn_type = self.buffer.explorer_input.default_reward_fn_type

        # check trainer_input.experience_buffer
        if self.mode == "both":
            if self.buffer.trainer_input.experience_buffer is None:
                self.buffer.trainer_input.experience_buffer = StorageConfig(
                    name="experience_buffer",
                    storage_type=StorageType.QUEUE,
                )
                logger.info(
                    f"Auto set `buffer.trainer_input.experience_buffer` to {self.buffer.trainer_input.experience_buffer}"
                )
        elif self.mode == "train":  # TODO: to be check
            if self.trainer.algorithm_type.is_dpo():
                if (
                    self.buffer.trainer_input.experience_buffer is None
                    or not self.buffer.trainer_input.experience_buffer.path
                ):
                    raise ValueError(
                        "`buffer.trainer_input.experience_buffer.path` is required when `trainer.algorithm_type == AlgorithmType.DPO`"
                    )
        if self.mode in ["both", "train"]:
            self.buffer.trainer_input.experience_buffer.algorithm_type = self.trainer.algorithm_type

        # set buffer.explorer_output
        if self.buffer.explorer_output is None:
            self.buffer.explorer_output = self.buffer.trainer_input.experience_buffer

        # check trainer_input.sft_warmup_dataset
        if (
            self.trainer.sft_warmup_steps > 0
            and self.buffer.trainer_input.sft_warmup_dataset is None
        ):
            raise ValueError(
                "buffer.trainer_input.sft_warmup_dataset is required when trainer.sft_warmup_steps > 0"
            )
        if self.buffer.trainer_input.sft_warmup_dataset is not None:
            self.buffer.trainer_input.sft_warmup_dataset.algorithm_type = AlgorithmType.SFT

        # set read_batch_size / pad_token_id / tokenizer_path
        self.buffer.read_batch_size = self.global_config.batch_size * self.explorer.repeat_times
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

    def check_and_update(self) -> None:  # noqa: C901
        """Check and update the config."""
        self._check_deprecated()

        # check mode
        if self.mode not in ["explore", "train", "both", "bench"]:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.trainer.algorithm_type == AlgorithmType.DPO and self.mode == "both":
            raise ValueError("DPO does not support `both` mode")

        # check model path
        if not os.path.isabs(self.model.checkpoint_path):
            self.model.checkpoint_path = os.path.join(os.getcwd(), self.model.checkpoint_path)
        if not self.model.critic_model_path:
            self.model.critic_model_path = self.model.model_path

        # check synchronizer
        self.synchronizer.explorer_world_size = (
            self.explorer.engine_num * self.explorer.tensor_parallel_size
        )
        self.synchronizer.backend = self.explorer.backend
        if self.mode == "bench" and self.synchronizer.sync_method != SyncMethod.CHECKPOINT:
            self.synchronizer.sync_method = SyncMethod.CHECKPOINT
            logger.warning(
                "Bench mode only supports checkpoint synchronization, set `synchronizer.sync_method` to `checkpoint`."
            )
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

        self._check_interval()

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
        if self.mode in {"both", "train"}:
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
