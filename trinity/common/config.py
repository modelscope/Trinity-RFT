# -*- coding: utf-8 -*-
"""Configs for RFT."""
from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import ray
from omegaconf import OmegaConf

from trinity.common.constants import (
    EXPLORER_NAME,
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    LOG_NODE_IP_ENV_VAR,
    PLUGIN_DIRS_ENV_VAR,
    TRAINER_NAME,
    PromptType,
    SaveStrategy,
    StorageType,
    SyncMethod,
    SyncStyle,
)
from trinity.utils.annotations import Experimental
from trinity.utils.log import get_logger
from trinity.utils.lora_utils import create_dummy_lora

logger = get_logger(__name__)


def set_if_none(obj, attr, val):
    if getattr(obj, attr, None) is None:
        setattr(obj, attr, val)


@dataclass
class FormatConfig:
    """Configuration for data formatting"""

    # for sft / dpo
    prompt_type: PromptType = PromptType.MESSAGES

    # for plaintext input
    prompt_key: str = "prompt"  # user prompt
    response_key: str = "response"  # assistant response
    system_prompt_key: Optional[str] = None  # If set, use the provided system prompt
    system_prompt: Optional[str] = None  # has lower priority than system_prompt_key

    # for message list input
    messages_key: str = "message"

    # for tools
    tools_key: str = "tools"
    image_key: Optional[str] = None  # used for multi-modal data
    video_key: Optional[str] = None  # used for multi-modal data

    reply_prefix: Optional[str] = None

    # for sample-level task controlling
    workflow_key: str = ""
    reward_fn_key: str = ""

    # for dpo dataset
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"

    # for multi-turn sft
    enable_concatenated_multi_turn: bool = False

    # for sft / dpo, if None, use model.custom_chat_template
    chat_template: Optional[str] = None


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    logprobs: int = 0  # vLLM return `logprobs + 1` elements
    max_tokens: Optional[int] = None  # if None, use model.max_response_tokens
    # repeat each task for `n` times
    # ! DO NOT SET in `buffer.explorer_input.taskset.rollout_args`
    n: int = 1


@dataclass
class OptimizerConfig:
    lr: float = 1e-6
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: Optional[float] = 0.0
    warmup_style: str = "constant"
    optimizer_type: str = "adam"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class LoRAConfig:
    """LoRA config, only effective for rollout model, not for auxiliary models."""

    name: Optional[str] = None
    path: Optional[str] = None
    base_model_name: Optional[str] = None
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dtype: str = "auto"
    target_modules: str = "all-linear"


@Experimental
@dataclass
class TaskSelectorConfig:
    """Data selector config."""

    selector_type: Optional[str] = "sequential"

    # For shuffle
    seed: int = 42

    # Estimator Config
    feature_keys: List[str] = field(default_factory=lambda: [])
    kwargs: dict = field(default_factory=dict)


@dataclass
class StorageConfig:
    """Storage config."""

    name: str = ""
    storage_type: StorageType = StorageType.FILE
    path: Optional[str] = None
    repeat_times: Optional[int] = None

    # For continuing training
    index: int = 0

    # used for StorageType.FILE
    split: str = "train"
    subset_name: Optional[str] = None
    format: FormatConfig = field(default_factory=FormatConfig)

    # used for StorageType.QUEUE
    capacity: int = 10000
    max_read_timeout: float = 1800
    use_priority_queue: bool = False
    reuse_cooldown_time: Optional[float] = None
    replay_buffer_kwargs: dict = field(
        default_factory=lambda: {"priority_fn": "linear_decay", "decay": 2.0}
    )

    # used for StorageType.SQL
    max_retry_times: int = 3
    max_retry_interval: int = 1

    # used for rollout tasks
    default_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None
    rollout_args: GenerationConfig = field(default_factory=GenerationConfig)
    workflow_args: dict = field(default_factory=dict)
    reward_fn_args: dict = field(default_factory=dict)
    task_selector: TaskSelectorConfig = field(default_factory=TaskSelectorConfig)

    # enable progress bar (tqdm) for _HFBatchReader
    enable_progress_bar: Optional[bool] = False

    # get storage from existing experiment
    ray_namespace: Optional[str] = None

    # ! DO NOT SET except you know what you are doing
    wrap_in_ray: bool = True

    # ! DO NOT SET, automatically set
    schema_type: Optional[str] = None

    # ! DO NOT SET, automatically set from buffer.total_epochs
    total_epochs: int = 1  # automatically set

    # ! DO NOT SET, automatically set from buffer.total_steps
    total_steps: Optional[int] = None  # automatically set

    # ! DO NOT SET,  automatically set corresponding to train/eval
    is_eval: bool = False


@dataclass
class OperatorConfig:
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


@Experimental
@dataclass
class ExperiencePipelineConfig:
    """Config for experience pipeline.

    Experience Pipeline is used to pre-process rollout experiences for better training.
    """

    # The list of experience operators to apply, operators will be applied in the order they are defined
    operators: List[OperatorConfig] = field(default_factory=list)
    save_input: bool = True  # whether to save the input experiences
    # the path to save the input experiences, can be a jsonl file or a sqlite database file
    input_save_path: Optional[str] = None

    # The following fields are experimental, do not set them unless you know what you are doing

    # A dictionary of input buffers, buffers are indexed by their names.
    # users only need to set extra buffers here
    inputs: Dict[str, StorageConfig] = field(default_factory=dict)
    # The output buffer will automatically set to the trainer input buffer, so we do not need to set it here.
    output: Optional[StorageConfig] = None


@Experimental
@dataclass
class TaskPipelineConfig:
    """Config for task pipeline.

    Task Pipeline is used to pre-process raw tasks for better exploring. Currently, we only support using
    Data-Juicer operators for task pipeline.
    """

    # The list of data-juicer operators to apply, operators will be applied in the order they are defined
    operators: List[OperatorConfig] = field(default_factory=list)
    # number of process
    num_process: int = 4
    # The path to the Data-Juicer config file. If set, operators and num_process will be ignored
    config_path: Optional[str] = None

    # Raw input tasksets. Currently, task pipeline only support local file as inputs,
    # e.g., /path/to/file.jsonl or /path/to/file.parquet, not a directory or huggingface path
    inputs: List[str] = field(default_factory=list)
    # Output task buffer, if not set, use `buffer.explorer_input.taskset`. In most cases, users do not need to set this field.
    output: Optional[StorageConfig] = None

    # The list of fields extracted from the input tasksets and processed into the output taskset
    target_fields: List[str] = field(default_factory=list)

    # weights for priority computing. Usually including 4 types of weights:
    # - difficulty
    # - diversity
    # - usage_frequency
    # - quality
    priority_weights: Dict[str, float] = field(default_factory=dict)

    # number of samples to select after task pipeline. -1 means all
    top_k: int = -1


@Experimental
@dataclass
class DataProcessorConfig:
    """Data Processor config"""

    # support two types of data pipelines for now
    # 1. For task. Data preprocessing from raw dataset to the task set
    task_pipeline: Optional[TaskPipelineConfig] = None
    # 2. For experience. Data processing for rollouts
    experience_pipeline: Optional[ExperiencePipelineConfig] = field(
        default_factory=ExperiencePipelineConfig
    )


@dataclass
class ModelConfig:
    # source model path
    model_path: str = ""
    critic_model_path: str = ""

    custom_chat_template: Optional[str] = None

    # the total number of tokens the model can handle
    max_model_len: Optional[int] = None

    # Note: the following fields are only for the `chat`/`generate` methods in `InferenceModel`
    # if you are using openai API, please set them when calling the API.

    # the maximum number of tokens for the prompt
    max_prompt_tokens: Optional[int] = None
    # the maximum number of tokens for the response
    max_response_tokens: Optional[int] = None
    # the minimum number of tokens for the response
    min_response_tokens: int = 1

    # lora config
    lora_configs: Optional[List[LoRAConfig]] = None
    fully_sharded_loras: bool = False
    max_cpu_loras: Optional[int] = None


@dataclass
class InferenceModelConfig:
    # ! DO NOT SET in explorer.rollout_model, automatically set from config.model.model_path
    model_path: str = ""

    engine_type: str = "vllm_async"
    engine_num: int = 1
    tensor_parallel_size: int = 1
    use_v1: bool = True
    enforce_eager: bool = True
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    seed: int = 42

    # if not set, use `model.max_model_len`
    max_model_len: Optional[int] = None
    # if not set, use `model.max_prompt_tokens`
    max_prompt_tokens: Optional[int] = None
    # if not set, use `model.max_response_tokens`
    max_response_tokens: Optional[int] = None
    # if not set, use `model.min_response_tokens`
    min_response_tokens: Optional[int] = None
    # used for testing very long response generation, do not set it unless you know what you are doing
    ignore_eos: bool = False

    # override chat template in model
    chat_template: Optional[str] = None

    # For Qwen3
    enable_thinking: bool = False

    # For history recording
    enable_history: bool = False

    # For OpenAI API
    enable_openai_api: bool = False

    # For tool calls in OpenAI API
    enable_auto_tool_choice: bool = False

    tool_call_parser: Optional[str] = None

    reasoning_parser: Optional[str] = None

    # ! DO NOT SET
    bundle_indices: str = ""

    # ! DO NOT SET, automatically set from model.lora_configs
    enable_lora: bool = False
    lora_modules: Optional[List[Dict]] = None
    lora_kwargs: Optional[dict] = field(default_factory=dict)


@dataclass
class AlgorithmConfig:
    """Config for algorithm."""

    algorithm_type: str = "ppo"
    # for GRPO-like algorithms, repeat each task for `repeat_times` times
    repeat_times: int = 1

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # the strategy for sampling experiences from the buffer
    sample_strategy: Optional[str] = None
    sample_strategy_args: Optional[dict] = None

    advantage_fn: Optional[str] = None  # "ppo"
    # If not set, use AdvantageFn.default_args()
    advantage_fn_args: Optional[dict] = None

    kl_penalty_fn: Optional[str] = None  # "none"  # set to "none" to disable kl penalty in reward
    # If not set, use kl_penalty_fn.default_args()
    kl_penalty_fn_args: Optional[dict] = None

    policy_loss_fn: Optional[str] = None  # "ppo"
    # If not set, use PolicyLossFn.default_args()
    policy_loss_fn_args: Optional[dict] = None

    kl_loss_fn: Optional[str] = None  # "k2"  # set to "none" to disable kl loss
    # If not set, use kl_loss_fn.default_args()
    kl_loss_fn_args: Optional[dict] = None

    entropy_loss_fn: Optional[str] = None  # "default"
    # If not set, use entropy_loss_fn.default_args()
    entropy_loss_fn_args: Optional[dict] = None


@dataclass
class ClusterConfig:
    """Config for the cluster."""

    ray_address: str = "auto"
    node_num: Optional[int] = None
    gpu_per_node: Optional[int] = None


@Experimental
@dataclass
class ExplorerInput:
    """Config for explorer input."""

    taskset: Optional[StorageConfig] = None
    tasksets: List[StorageConfig] = field(default_factory=list)
    eval_tasksets: List[StorageConfig] = field(default_factory=list)
    # The following args provide default values for the corresponding args in `taskset` and `eval_tasksets`
    default_workflow_type: Optional[str] = None
    default_eval_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None


@dataclass
class TrainerInput:
    """Config for trainer input."""

    # The main experience buffer to be used in trainer
    # Commonly, it is also the output buffer of the Explorer
    experience_buffer: Optional[StorageConfig] = None

    # Some auxiliary buffers to facilitate training (e.g., data mixing)
    auxiliary_buffers: Dict[str, StorageConfig] = field(default_factory=dict)


@dataclass
class BufferConfig:
    """Config for buffer."""

    batch_size: int = 1
    train_batch_size: int = 0  # default to `batch_size` * `algorithm.n`
    total_epochs: int = 1
    total_steps: Optional[int] = None

    # for explorer
    explorer_input: ExplorerInput = field(default_factory=ExplorerInput)

    # for trainer
    trainer_input: TrainerInput = field(default_factory=TrainerInput)

    # ! DO NOT SET FOLLOWING FIELDS
    explorer_output: Optional[StorageConfig] = None  # automatically set
    tokenizer_path: Optional[str] = None  # automatically set
    pad_token_id: Optional[int] = None  # automatically set
    cache_dir: Optional[str] = None  # automatically set


@dataclass
class ExplorerConfig:
    """Config for explorer."""

    name: str = EXPLORER_NAME
    # for workflow runner
    # number of workflow runners.
    runner_per_model: int = 8  # number of runners per each rollout model
    max_timeout: int = 1800  # wait each task for 30 minutes
    max_retry_times: int = 2  # retry each task for 2 times if it fails or timeout
    env_vars: dict = field(default_factory=dict)  # environment variables for workflow runner
    max_repeat_times_per_runner: Optional[
        int
    ] = None  # the number of time to repeat each task in a single workflow runner (for GRPO-like algorithms)

    runner_num: Optional[int] = None  # ! Deprecated

    # for inference models
    # for rollout model
    rollout_model: InferenceModelConfig = field(default_factory=InferenceModelConfig)
    # for other models used in the custom workflows
    auxiliary_models: List[InferenceModelConfig] = field(default_factory=list)

    # for evaluation
    eval_interval: int = 100
    eval_on_startup: bool = True  # evalulate at step 0

    # for benchmark
    bench_on_latest_checkpoint: bool = False  # only benchmark the latest checkpoint

    # for serve mode
    api_port: int = 8010
    # listen on all interfaces by default
    listen_address: str = "0.0.0.0"
    # check the running status of the server every 60 seconds
    service_status_check_interval: int = 60
    # keep at least 1 model in running status
    min_running_model_num: int = 1


@dataclass
class TrainerConfig:
    name: str = TRAINER_NAME
    trainer_type: str = "verl"
    save_interval: int = 0
    enable_preview: bool = True  # enable rollout preview in wandb
    total_steps: Optional[
        int
    ] = None  # total training steps, training stops when reaching this step, None means no limit

    # trainer configs
    grad_clip: float = 1.0
    use_dynamic_bsz: bool = True
    # if None, automatically set to ceil(2 * model.max_model_len / ulysses_sequence_parallel_size)
    max_token_len_per_gpu: Optional[int] = None
    ulysses_sequence_parallel_size: int = 1  # sp size
    # TODO: extract more train-related params from underlying trainer engine

    save_strategy: SaveStrategy = SaveStrategy.UNRESTRICTED

    trainer_config: Any = field(default_factory=dict)
    trainer_config_path: str = ""  # deprecated, use `trainer_config` instead


@dataclass
class MonitorConfig:
    # TODO: support multiple monitors (List[str])
    monitor_type: str = "tensorboard"
    # the default args for monitor
    monitor_args: Optional[Dict] = None
    # whether to enable ray timeline profile
    # the output file will be saved to `cache_dir/timeline.json`
    enable_ray_timeline: bool = False
    # ! DO NOT SET, automatically generated as checkpoint_job_dir/monitor
    cache_dir: str = ""


@dataclass
class SynchronizerConfig:
    """Configs for model weight synchronization."""

    sync_method: SyncMethod = SyncMethod.NCCL
    sync_style: SyncStyle = SyncStyle.FIXED
    # sync weights every `sync_interval` steps
    sync_interval: int = 1
    # allow explorer to run `sync_offset` steps before sync
    sync_offset: int = 0
    # waiting for `sync_timeout` seconds before timeout in `nccl` method
    sync_timeout: int = 3600
    # wait for the lastest checkpoint to be ready  # TODO: to be used
    wait_for_checkpoint: bool = False

    # ! DO NOT SET, automatically calculated
    explorer_world_size: Optional[int] = None
    ray_namespace: str = ""


@dataclass
class DataJuicerServiceConfig:
    """Config for Data-Juicer.

    Please update `trinity.service.data_juicer.server.server.py` correspondingly if you change the fields here.
    """

    # the url of the Data-Juicer server
    server_url: Optional[str] = None

    # whether to start Data-Juicer server automatically
    auto_start: bool = False

    # the following fields are only used when `auto_start` is True
    # the port of the Data-Juicer server, if not set, a random port will be used
    port: Optional[int] = None
    # the hostname will be automatically set to "localhost" so we do not need to set it here


@dataclass
class ServiceConfig:
    """Configs for outside services."""

    data_juicer: Optional[DataJuicerServiceConfig] = None


@dataclass
class LogConfig:
    """Configs for logger."""

    level: str = "INFO"  # default log level (DEBUG, INFO, WARNING, ERROR)
    group_by_node: bool = False  # whether to group logs by node IP in Ray cluster
    # ! DO NOT SET, automatically generated as <checkpoint_root_dir>/<project>/<name>/log
    save_dir: str = ""


@dataclass
class StageConfig:
    """Configs for a stage."""

    stage_name: str
    mode: Optional[str] = None
    algorithm: Optional[AlgorithmConfig] = None
    buffer: Optional[BufferConfig] = None
    data_processor: Optional[DataProcessorConfig] = None
    explorer: Optional[ExplorerConfig] = None
    trainer: Optional[TrainerConfig] = None


@dataclass
class Config:
    """Global Configuration"""

    mode: str = "both"  # `explore`, `train`, `both` or `bench`
    project: str = "Trinity-RFT"
    group: str = ""
    name: str = "rft"
    # the root dir for checkpoints
    checkpoint_root_dir: str = ""
    # ! DO NOT SET, automatically generated as `checkpoint_root_dir/project/name`
    checkpoint_job_dir: str = ""
    # If not set, automatically generated as f"{config.project}-{config.name}"
    ray_namespace: str = ""
    # whether to continue training from the last checkpoint in checkpoint_job_dir (if any)
    continue_from_checkpoint: bool = True

    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    data_processor: DataProcessorConfig = field(default_factory=DataProcessorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    explorer: ExplorerConfig = field(default_factory=ExplorerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    synchronizer: SynchronizerConfig = field(default_factory=SynchronizerConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # configurations for different training stages
    stages: List[StageConfig] = field(default_factory=list)

    def save(self, config_path: str) -> None:
        """Save config to file."""
        with open(config_path, "w", encoding="utf-8") as f:
            OmegaConf.save(self, f)

    def _check_deprecated(self) -> None:
        if self.explorer.runner_num is not None:
            logger.warning(
                "`explorer.runner_num` is deprecated, please use `explorer.runner_per_model` instead."
            )

    def _update_config_from_ray_cluster(self) -> None:
        """Update config if `node_num` or `gpu_per_node` are not set."""
        if self.cluster.node_num is not None and self.cluster.gpu_per_node is not None:
            return

        # init ray cluster to detect node_num and gpu_per_node
        was_initialized = ray.is_initialized()
        if not was_initialized:
            ray.init(
                address=self.cluster.ray_address,
                ignore_reinit_error=True,
                namespace=self.ray_namespace,
            )

        alive_nodes = [n for n in ray.nodes() if n["alive"]]
        if not alive_nodes:
            raise RuntimeError("Could not find any alive nodes in the Ray cluster.")

        # set node_num
        if self.cluster.node_num is None:
            self.cluster.node_num = len(alive_nodes)
            logger.info(f"Auto-detected and set node_num: {self.cluster.node_num}")

        # set gpu_per_node
        if self.cluster.gpu_per_node is None:
            gpu_per_node = 0
            for node in alive_nodes:
                node_gpus = node.get("Resources", {}).get("GPU")
                if node_gpus and node_gpus > 0:
                    gpu_per_node = int(node_gpus)
                    break

            self.cluster.gpu_per_node = gpu_per_node
            logger.info(f"Auto-detected and set gpu_per_node: {self.cluster.gpu_per_node}")

        if not was_initialized:
            ray.shutdown()

    def _check_interval(self) -> None:
        assert self.synchronizer.sync_interval > 0

        if self.mode != "bench" and self.algorithm.algorithm_type != "dpo":  # TODO
            # check eval_interval
            if self.explorer.eval_interval % self.synchronizer.sync_interval != 0:
                self.explorer.eval_interval = (
                    max(self.explorer.eval_interval // self.synchronizer.sync_interval, 1)
                ) * self.synchronizer.sync_interval
                logger.warning(
                    f"`eval_interval` is not a multiple of `sync_interval`; adjusted to the nearest integer={self.explorer.eval_interval}."
                )

    def _check_buffer(self) -> None:  # noqa: C901
        # TODO: split this function into different buffer read/writer
        # check explorer_input
        trainer_input = self.buffer.trainer_input
        experience_buffer = trainer_input.experience_buffer
        explorer_input = self.buffer.explorer_input

        if explorer_input.taskset:
            if len(explorer_input.tasksets) > 0:
                raise ValueError("Do not support setting `taskset` and `tasksets` simultaneously!")
            explorer_input.tasksets = [explorer_input.taskset]
            explorer_input.taskset = None
        else:
            if len(explorer_input.tasksets) == 0:
                explorer_input.tasksets = [StorageConfig()]
        tasksets = explorer_input.tasksets

        if self.mode == "train":
            assert (
                experience_buffer is not None
            ), "`buffer.trainer_input.experience_buffer` is required when `mode` is `train`."
            experience_buffer.total_epochs = self.buffer.total_epochs
            experience_buffer.total_steps = self.buffer.total_steps

        for taskset in tasksets:
            if self.mode != "train" and not taskset.path:
                raise ValueError(
                    "`buffer.explorer_input.taskset.path` is required, please set it to the path of the taskset."
                )
            if not taskset.name:
                taskset.name = "taskset"
            if taskset.repeat_times is None or taskset.repeat_times != self.algorithm.repeat_times:
                taskset.repeat_times = self.algorithm.repeat_times
                logger.info(
                    "`buffer.explorer_input.taskset.repeat_times` is set to `algorithm.repeat_times`"
                    f" (={self.algorithm.repeat_times})."
                )
            taskset.total_epochs = self.buffer.total_epochs
            taskset.total_steps = self.buffer.total_steps

            set_if_none(taskset, "default_workflow_type", explorer_input.default_workflow_type)
            set_if_none(taskset, "default_reward_fn_type", explorer_input.default_reward_fn_type)
            set_if_none(taskset, "ray_namespace", self.ray_namespace)
            set_if_none(taskset.rollout_args, "max_tokens", self.model.max_response_tokens)

        remained_tasksets = []
        for idx, dataset in enumerate(explorer_input.eval_tasksets):
            if not dataset.path:
                logger.warning(f"Eval dataset [{dataset}]'s path is not configured. Skip.")
                continue
            dataset.is_eval = True
            if not dataset.name:
                dataset.name = f"eval_taskset_{idx}"
            set_if_none(dataset, "repeat_times", 1)
            # eval_workflow has higher priority than workflow in eval tasksets, so we set it first
            set_if_none(dataset, "default_workflow_type", explorer_input.default_eval_workflow_type)
            set_if_none(dataset, "default_workflow_type", explorer_input.default_workflow_type)
            set_if_none(dataset, "default_reward_fn_type", explorer_input.default_reward_fn_type)
            set_if_none(dataset, "ray_namespace", self.ray_namespace)
            set_if_none(dataset.rollout_args, "max_tokens", self.model.max_response_tokens)
            remained_tasksets.append(dataset)
        explorer_input.eval_tasksets = remained_tasksets

        # check trainer_input.experience_buffer
        if experience_buffer is None:
            experience_buffer = trainer_input.experience_buffer = StorageConfig(
                name="experience_buffer",
                storage_type=StorageType.QUEUE,
            )
            logger.info(f"Auto set `buffer.trainer_input.experience_buffer` to {experience_buffer}")
        elif experience_buffer.storage_type is StorageType.FILE and self.mode == "both":
            logger.warning(
                "`FILE` storage is not supported to use as experience_buffer in `both` mode, use `QUEUE` instead."
            )
            experience_buffer.storage_type = StorageType.QUEUE

        from trinity.algorithm.algorithm import ALGORITHM_TYPE

        experience_buffer.schema_type = ALGORITHM_TYPE.get(self.algorithm.algorithm_type).schema

        set_if_none(experience_buffer, "ray_namespace", self.ray_namespace)
        set_if_none(experience_buffer.format, "chat_template", self.model.custom_chat_template)

        # create buffer.cache_dir at <checkpoint_root_dir>/<project>/<name>/buffer
        self.buffer.cache_dir = os.path.abspath(os.path.join(self.checkpoint_job_dir, "buffer"))
        try:
            os.makedirs(self.buffer.cache_dir, exist_ok=True)
        except Exception:
            logger.warning(
                f"Failed to create buffer dir {self.buffer.cache_dir}, please check "
                f"your checkpoint directory: {self.checkpoint_job_dir}"
            )

        # check input/output buffers in pipelines
        experience_pipeline = self.data_processor.experience_pipeline
        if experience_pipeline is not None:
            if experience_pipeline.save_input and experience_pipeline.input_save_path is None:
                experience_pipeline.input_save_path = os.path.join(
                    self.buffer.cache_dir, "explorer_output.jsonl"
                )
                logger.info(
                    f"Auto set `data_processor.experience_pipeline.input_save_path` to {experience_pipeline.input_save_path}"
                )

        task_pipeline = self.data_processor.task_pipeline
        if task_pipeline is not None:
            if task_pipeline.output is None:
                if tasksets and tasksets[0].path is not None:
                    task_pipeline.output = tasksets[0]
                elif (
                    experience_buffer.schema_type in {"dpo", "sft"}
                    and experience_buffer.path is not None
                ):
                    task_pipeline.output = experience_buffer
                else:
                    raise ValueError(
                        "`data_processor.task_pipeline.output` is required when both "
                        "`buffer.explorer_input.tasksets[0].path` and `buffer.trainer_input.experience_buffer.path` are "
                        "None"
                    )
            if task_pipeline.output.path and os.path.exists(task_pipeline.output.path):
                raise ValueError(
                    f"Task pipeline output path {task_pipeline.output.path} already exists.\n"
                    "Please choose a different output path to avoid overwriting."
                )

        # check train_batch_size
        if not self.buffer.train_batch_size:
            if self.mode == "train" or self.algorithm.algorithm_type in ["sft", "dpo"]:
                raise ValueError(
                    "`buffer.train_batch_size` is required when `mode` is 'train' or `algorithm.algorithm_type` is "
                    "'sft' or 'dpo'"
                )
            logger.info(
                "`buffer.train_batch_size` is set to `buffer.batch_size` * `algorithm.repeat_times`"
            )
            self.buffer.train_batch_size = self.buffer.batch_size * self.algorithm.repeat_times

        # set pad_token_id / tokenizer_path
        if self.buffer.pad_token_id is None:
            from transformers import AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model.model_path)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    logger.warning(
                        f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}",
                        stacklevel=1,
                    )
                self.buffer.pad_token_id = tokenizer.pad_token_id

            except Exception:
                logger.warning(f"Failed to get pad token id from model {self.model.model_path}")
                self.buffer.pad_token_id = 0
        self.buffer.tokenizer_path = self.model.model_path

    def _check_algorithm(self) -> None:
        from trinity.algorithm import (
            ADVANTAGE_FN,
            ENTROPY_LOSS_FN,
            KL_FN,
            POLICY_LOSS_FN,
            SAMPLE_STRATEGY,
        )
        from trinity.algorithm.algorithm import ALGORITHM_TYPE

        algorithm = ALGORITHM_TYPE.get(self.algorithm.algorithm_type)
        algorithm.check_config(self)
        default_config = {
            "sample_strategy": "warmup",
            "policy_loss_fn": "ppo",
            "advantage_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }
        default_config.update(algorithm.default_config())
        for key, value in default_config.items():
            set_if_none(self.algorithm, key, value)

        def check_and_set(name, registry, args_attr):
            fn_cls = registry.get(getattr(self.algorithm, name))
            if fn_cls is None:
                raise ValueError(f"Invalid {name}: {getattr(self.algorithm, name)}")
            set_if_none(self.algorithm, args_attr, fn_cls.default_args())
            return fn_cls

        check_and_set("sample_strategy", SAMPLE_STRATEGY, "sample_strategy_args")
        check_and_set("policy_loss_fn", POLICY_LOSS_FN, "policy_loss_fn_args")
        check_and_set("advantage_fn", ADVANTAGE_FN, "advantage_fn_args")
        check_and_set("kl_loss_fn", KL_FN, "kl_loss_fn_args")
        check_and_set("kl_penalty_fn", KL_FN, "kl_penalty_fn_args")
        check_and_set("entropy_loss_fn", ENTROPY_LOSS_FN, "entropy_loss_fn_args")

    def _check_model(self) -> None:
        model = self.model
        if not model.critic_model_path:
            model.critic_model_path = model.model_path

        # check max_model_len, max_prompt_tokens, max_response_tokens

        # if all three are set, check if they are valid
        if (
            model.max_model_len is not None
            and model.max_prompt_tokens is not None
            and model.max_response_tokens is not None
        ):
            if model.max_prompt_tokens + model.max_response_tokens > model.max_model_len:
                raise ValueError(
                    f"`max_prompt_tokens` + `max_response_tokens` ({model.max_prompt_tokens} + {model.max_response_tokens}) "
                    f"exceeds `max_model_len` ({model.max_model_len}). Please adjust them accordingly."
                )

        # check max_model_len first
        if model.max_model_len is None:
            if model.max_prompt_tokens is not None and model.max_response_tokens is not None:
                model.max_model_len = model.max_prompt_tokens + model.max_response_tokens
                logger.warning(
                    f"`max_model_len` is set to {model.max_model_len} from `max_prompt_tokens` and `max_response_tokens`."
                )
            else:
                raise ValueError("Unable to determine `max_model_len`, please set it manually.")

        # both max_prompt_tokens and max_response_tokens are None
        if model.max_prompt_tokens is None and model.max_response_tokens is None:
            # default to max_model_len / 2
            model.max_prompt_tokens = model.max_model_len // 2
            model.max_response_tokens = model.max_model_len - model.max_prompt_tokens
            logger.warning(
                f"`max_prompt_tokens` and `max_response_tokens` are not set, set to {model.max_prompt_tokens} and {model.max_response_tokens} respectively."
            )

        # only max_prompt_tokens is None
        if model.max_prompt_tokens is None and model.max_response_tokens is not None:
            model.max_response_tokens = min(model.max_response_tokens, model.max_model_len - 1)
            model.max_prompt_tokens = model.max_model_len - model.max_response_tokens
            logger.warning(
                f"`max_prompt_tokens` is set to {model.max_prompt_tokens}, `max_response_tokens` is set to {model.max_response_tokens}."
            )

        # only max_response_tokens is None
        if model.max_response_tokens is None and model.max_prompt_tokens is not None:
            model.max_prompt_tokens = min(model.max_prompt_tokens, model.max_model_len - 1)
            model.max_response_tokens = model.max_model_len - model.max_prompt_tokens
            logger.warning(
                f"`max_response_tokens` is set to {model.max_response_tokens}, `max_prompt_tokens` is set to {model.max_prompt_tokens}."
            )

        if model.min_response_tokens >= model.max_response_tokens:  # type: ignore [operator]
            model.min_response_tokens = max(model.max_response_tokens - 1, 0)  # type: ignore [operator]
            logger.warning(f"`min_response_tokens` is set to {model.min_response_tokens}.")

    def __iter__(self):
        """Iterate over configs with each stage applied in order.

        Yields:
            Config: The config after applying each stage.
        """
        for stage in self.stages:
            new_config = deepcopy(self)
            for field_name in stage.__dataclass_fields__:
                stage_value = getattr(stage, field_name)
                if stage_value is not None and hasattr(new_config, field_name):
                    setattr(new_config, field_name, stage_value)
            if stage.stage_name:
                new_config.name = f"{self.name}/{stage.stage_name}"
            new_config.stages = []
            yield new_config

    def check_and_update(self) -> Config:  # noqa: C901
        """Check and update the config."""
        self._check_deprecated()

        # set namespace
        if self.ray_namespace is None or len(self.ray_namespace) == 0:
            self.ray_namespace = f"{self.project}/{self.name}"

        # check cluster infomation
        self._update_config_from_ray_cluster()

        # check algorithm
        self._check_algorithm()

        # check mode
        if self.mode not in ["explore", "train", "both", "bench", "serve"]:
            raise ValueError(f"Invalid mode: {self.mode}")

        # prepare for the checkpoint directory
        if not os.path.isabs(self.checkpoint_root_dir):
            self.checkpoint_root_dir = os.path.join(os.getcwd(), self.checkpoint_root_dir)
        # create a job dir at checkpoint_root_dir/project/name
        self.checkpoint_job_dir = os.path.join(
            self.checkpoint_root_dir, self.project, self.group, self.name
        )
        # rename the experiment when necessary
        if not self.continue_from_checkpoint and (
            os.path.exists(self.checkpoint_job_dir) and os.listdir(self.checkpoint_job_dir)
        ):
            if self.mode == "bench":
                logger.warning(
                    "For bench mode, `continue_from_checkpoint` is set as `true` to enable using existing checkpoints."
                )
                self.continue_from_checkpoint = True
            else:
                ori_name = self.name
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                self.name = f"{ori_name}_{timestamp}"
                self.checkpoint_job_dir = f"{self.checkpoint_job_dir}_{timestamp}"
                logger.warning(f"Experiment [{ori_name}] already exists, renamed as {self.name}.")
        os.makedirs(self.checkpoint_job_dir, exist_ok=True)

        # check model
        self._check_model()

        # check explorer
        if self.explorer is not None:
            self.explorer.rollout_model.model_path = self.model.model_path
            self.explorer.rollout_model.max_model_len = self.model.max_model_len
            self.explorer.rollout_model.max_prompt_tokens = self.model.max_prompt_tokens
            self.explorer.rollout_model.max_response_tokens = self.model.max_response_tokens
            self.explorer.rollout_model.min_response_tokens = self.model.min_response_tokens
            for aux_model in self.explorer.auxiliary_models:
                if not aux_model.model_path:
                    raise ValueError("auxiliary model's model_path is required.")
                set_if_none(aux_model, "max_model_len", self.model.max_model_len)
                set_if_none(aux_model, "max_prompt_tokens", self.model.max_prompt_tokens)
                set_if_none(aux_model, "max_response_tokens", self.model.max_response_tokens)
                set_if_none(aux_model, "min_response_tokens", self.model.min_response_tokens)

            # for lora configs
            if self.model.lora_configs is not None:
                self.explorer.rollout_model.enable_lora = True
                if len(self.model.lora_configs) > 1:
                    raise ValueError("Only one lora adapter is supported for now.")
                if self.model.lora_configs[0].path is None:
                    logger.info("Creating dummy lora, since no lora_path is provided.")
                    lora_path = create_dummy_lora(
                        model_path=self.model.model_path,
                        checkpoint_job_dir=self.checkpoint_job_dir,
                        lora_rank=self.model.lora_configs[0].lora_rank,
                        lora_alpha=self.model.lora_configs[0].lora_alpha,
                        target_modules=self.model.lora_configs[0].target_modules,
                    )
                    self.model.lora_configs[0].path = lora_path
                self.explorer.rollout_model.lora_modules = [
                    {
                        "lora_int_id": i + 1,
                        "lora_name": cfg.name,
                        "lora_path": cfg.path,
                        "base_model_name": cfg.base_model_name,
                    }
                    for i, cfg in enumerate(self.model.lora_configs)
                ]
                self.explorer.rollout_model.lora_kwargs = {
                    "max_loras": len(self.model.lora_configs),
                    "max_lora_rank": max(
                        (
                            model_config.lora_rank
                            for model_config in self.model.lora_configs
                            if model_config.lora_rank > 0
                        ),
                        default=0,
                    ),
                    "default_lora_path": os.path.join(
                        self.checkpoint_job_dir, "global_step_0", "actor", "lora_adapter"
                    ),  # will be poped later
                }

        # check synchronizer
        self.synchronizer.ray_namespace = self.ray_namespace
        self.synchronizer.explorer_world_size = (
            self.explorer.rollout_model.engine_num
            * self.explorer.rollout_model.tensor_parallel_size
        )
        if (
            self.mode in ["train", "explore", "bench", "serve"]
            and self.synchronizer.sync_method == SyncMethod.NCCL
        ):
            self.synchronizer.sync_method = SyncMethod.CHECKPOINT
            logger.warning(
                f"`{self.mode}` mode does not support NCCL synchronization, set `synchronizer.sync_method` to `checkpoint`."
            )

        self._check_interval()

        # check monitor
        from trinity.utils.monitor import MONITOR

        monitor_cls = MONITOR.get(self.monitor.monitor_type)
        if monitor_cls is None:
            raise ValueError(f"Invalid monitor type: {self.monitor.monitor_type}")
        set_if_none(self.monitor, "monitor_args", monitor_cls.default_args())
        # create a job dir in <checkpoint_root_dir>/<project>/<name>/monitor
        self.monitor.cache_dir = os.path.join(self.checkpoint_job_dir, "monitor")
        try:
            os.makedirs(self.monitor.cache_dir, exist_ok=True)
        except Exception:
            logger.warning(
                f"Failed to create monitor dir {self.monitor.cache_dir}, please check "
                f"your checkpoint directory: {self.checkpoint_job_dir}"
            )

        # check buffer
        self._check_buffer()
        # check and update trainer
        if self.mode in ["train", "both", "bench"]:
            if self.trainer.trainer_type == "verl":
                if self.trainer.trainer_config:
                    from trinity.common.verl_config import veRLConfig

                    trainer_config_schema = OmegaConf.structured(veRLConfig)
                    trainer_config = OmegaConf.merge(
                        trainer_config_schema, self.trainer.trainer_config
                    )
                    self.trainer.trainer_config = OmegaConf.to_object(trainer_config)
                elif self.trainer.trainer_config_path:
                    raise ValueError(
                        "`trainer_config_path` is deprecated; please use `trainer_config` instead."
                    )
                else:
                    from trinity.common.verl_config import veRLConfig

                    logger.info("`trainer_config` is not provided, using default trainer config.")
                    self.trainer.trainer_config = veRLConfig()
                if self.trainer.max_token_len_per_gpu is None:
                    self.trainer.max_token_len_per_gpu = math.ceil(
                        2 * self.model.max_model_len / self.trainer.ulysses_sequence_parallel_size  # type: ignore [operator]
                    )
            else:
                raise ValueError(f"Invalid trainer type: {self.trainer_type}")
            self.trainer.trainer_config.synchronize_config(self)

        # check service
        if self.service.data_juicer is not None:
            for operator in self.data_processor.experience_pipeline.operators:
                if operator.name == "data_juicer":
                    operator.args["service_config"] = self.service.data_juicer

        # check log
        self.log.save_dir = os.path.join(self.checkpoint_job_dir, "log")
        return self

    def flatten(self) -> Dict[str, Any]:
        """Flatten the config into a single-level dict with dot-separated keys for nested fields."""

        def _flatten(obj, parent_key="", sep="."):
            items = {}
            if hasattr(obj, "__dataclass_fields__"):
                obj = vars(obj)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    items.update(_flatten(v, new_key, sep=sep))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                    items.update(_flatten(v, new_key, sep=sep))
            elif isinstance(obj, Enum):
                items[parent_key] = obj.value
            else:
                items[parent_key] = obj
            return items

        return _flatten(self)

    def get_envs(self) -> Dict[str, str]:
        """Get the environment variables from the config."""
        return {
            PLUGIN_DIRS_ENV_VAR: os.getenv(PLUGIN_DIRS_ENV_VAR, ""),
            LOG_LEVEL_ENV_VAR: self.log.level,
            LOG_DIR_ENV_VAR: self.log.save_dir,
            LOG_NODE_IP_ENV_VAR: "1" if self.log.group_by_node else "0",
        }


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
