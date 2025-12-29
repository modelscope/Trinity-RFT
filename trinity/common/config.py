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
    temperature: Optional[float] = None  # 1.0
    top_p: Optional[float] = None  # 1.0
    top_k: int = -1  # -1 means disabled
    logprobs: Optional[int] = None  # 0  # vLLM return `logprobs + 1` elements
    max_tokens: Optional[int] = None  # if None, use model.max_response_tokens
    # repeat each task for `n` times
    # ! DO NOT SET, it will be set by `algorithm.repeat_times` or `buffer.explorer_input.eval_tasksets[i].repeat_times`
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
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "constant"
    min_lr: float = 0.0


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
class ReplayBufferConfig:
    """Config for replay buffer used in StorageType.QUEUE."""

    enable: bool = False
    priority_fn: str = "linear_decay"
    reuse_cooldown_time: Optional[float] = None
    priority_fn_args: Dict = field(default_factory=lambda: {"decay": 2.0})


@dataclass
class OverRolloutConfig:
    """Config for over-rollout in explorer."""

    ratio: float = 0.0  # explorer will only wait for (1 - over_rollout.ratio) * batch_size of tasks at each step
    wait_after_min: float = 30.0  # wait 30 s after reaching minimum task threshold
    # more settings will be added in the future
    # e.g., postpone tasks into the next step if not finished in time


@dataclass
class DynamicTimeoutConfig:
    """Config for dynamic timeout in explorer."""

    enable: bool = False
    ratio: float = 3.0  # the timeout for each step will be min(max_timeout, average_time_per_task * dynamic_timeout.ratio)


@dataclass
class StorageConfig:
    """Storage config for both taskset and experience buffer.
    Not visible to users directly. Please use ExperienceBufferConfig or TasksetConfig instead.
    """

    name: str = ""
    storage_type: str = StorageType.FILE.value
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
    replay_buffer: Optional[ReplayBufferConfig] = field(default_factory=ReplayBufferConfig)

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

    # ! DO NOT SET, automatically set from buffer.batch_size / train_batch_size
    batch_size: int = 0

    # ! DO NOT SET, automatically set from model.model_path
    tokenizer_path: Optional[str] = None

    # ! DO NOT SET,  automatically set corresponding to train/eval
    is_eval: bool = False


@dataclass
class TasksetConfig:
    name: str = ""
    storage_type: str = StorageType.FILE.value
    path: Optional[str] = None

    default_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None
    rollout_args: GenerationConfig = field(default_factory=GenerationConfig)
    workflow_args: dict = field(default_factory=dict)
    reward_fn_args: dict = field(default_factory=dict)
    task_selector: TaskSelectorConfig = field(default_factory=TaskSelectorConfig)

    # used for StorageType.FILE
    split: str = "train"
    subset_name: Optional[str] = None
    format: FormatConfig = field(default_factory=FormatConfig)

    # used for StorageType.SQL
    max_retry_times: int = 3
    max_retry_interval: int = 1

    enable_progress_bar: bool = False

    # ! This setting is only valid for `eval_taskset`; for other taskset, it will be overridden by `algorithm.repeat_times`.
    repeat_times: int = 1
    # ! DO NOT SET, automatically load from checkpoint
    index: int = 0
    # ! DO NOT SET, automatically set based on train/eval
    is_eval: bool = False
    # ! DO NOT SET, automatically set from buffer.batch_size
    batch_size: int = 0
    # ! DO NOT SET, automatically set from buffer.total_epochs
    total_epochs: int = 1  # automatically set
    # ! DO NOT SET, automatically set from buffer.total_steps
    total_steps: Optional[int] = None  # automatically set

    def to_storage_config(self) -> StorageConfig:
        storage_config = StorageConfig(
            name=self.name,
            storage_type=self.storage_type,
            path=self.path,
            task_selector=self.task_selector,
            repeat_times=self.repeat_times,
            split=self.split,
            subset_name=self.subset_name,
            format=self.format,
            max_retry_times=self.max_retry_times,
            max_retry_interval=self.max_retry_interval,
            default_workflow_type=self.default_workflow_type,
            default_reward_fn_type=self.default_reward_fn_type,
            rollout_args=self.rollout_args,
            workflow_args=self.workflow_args,
            reward_fn_args=self.reward_fn_args,
            enable_progress_bar=self.enable_progress_bar,
            index=self.index,
            is_eval=self.is_eval,
            batch_size=self.batch_size,
            total_epochs=self.total_epochs,
            total_steps=self.total_steps,
        )
        return storage_config


@dataclass
class ExperienceBufferConfig:
    """Storage Config for trainer input experience buffer."""

    name: str = ""
    storage_type: str = StorageType.QUEUE.value
    path: Optional[str] = None

    # used for StorageType.QUEUE
    capacity: int = 10000
    max_read_timeout: float = 1800
    replay_buffer: Optional[ReplayBufferConfig] = field(default_factory=ReplayBufferConfig)

    # used for StorageType.SQL
    max_retry_times: int = 3
    max_retry_interval: int = 1

    # used for StorageType.FILE
    split: str = "train"
    subset_name: Optional[str] = None
    format: FormatConfig = field(default_factory=FormatConfig)
    enable_progress_bar: Optional[bool] = False

    # ! DO NOT SET, automatically set
    schema_type: Optional[str] = None
    # ! DO NOT SET
    index: int = 0
    # ! DO NOT SET, automatically set from buffer.batch_size
    batch_size: int = 0
    # ! DO NOT SET, automatically set from model.model_path
    tokenizer_path: Optional[str] = None
    # ! DO NOT SET, automatically set from buffer.total_epochs
    total_epochs: int = 1  # automatically set
    # ! DO NOT SET, automatically set from buffer.total_steps
    total_steps: Optional[int] = None  # automatically set

    def to_storage_config(self) -> StorageConfig:
        storage_config = StorageConfig(
            name=self.name,
            storage_type=self.storage_type,
            path=self.path,
            capacity=self.capacity,
            max_read_timeout=self.max_read_timeout,
            replay_buffer=self.replay_buffer,
            max_retry_times=self.max_retry_times,
            max_retry_interval=self.max_retry_interval,
            split=self.split,
            subset_name=self.subset_name,
            format=self.format,
            enable_progress_bar=self.enable_progress_bar,
            schema_type=self.schema_type,
            index=self.index,
            batch_size=self.batch_size,
            tokenizer_path=self.tokenizer_path,
            total_epochs=self.total_epochs,
            total_steps=self.total_steps,
        )
        return storage_config


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
    inputs: Dict[str, ExperienceBufferConfig] = field(default_factory=dict)
    # The output buffer will automatically set to the trainer input buffer, so we do not need to set it here.
    output: Optional[ExperienceBufferConfig] = None


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
    output: Optional[TasksetConfig] = None

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
class TinkerConfig:
    enable: bool = False
    rank: int = 32  # lora rank
    seed: Optional[int] = None
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True


@dataclass
class ModelConfig:
    # source model path
    model_path: str = ""
    critic_model_path: str = ""

    custom_chat_template: Optional[str] = None
    chat_template_path: Optional[
        str
    ] = None  # path to the chat template file, e.g., jinja2 type; overrides `custom_chat_template` if set

    # rollout args
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    logprobs: int = 0

    # the total number of tokens the model can handle
    max_model_len: Optional[int] = None

    # Note: the following fields are only for the `chat`/`generate` methods in `InferenceModel`
    # if you are using openai API, please set them when calling the API.

    # the maximum number of tokens for the prompt
    max_prompt_tokens: Optional[int] = None
    # the maximum number of tokens for the response
    max_response_tokens: Optional[int] = None
    # the minimum number of tokens for the response
    min_response_tokens: int = 0
    # whether to truncate the prompt; if set to True, the prompt will be truncated to `max_prompt_tokens` tokens;
    # not applicable for OpenAI API
    enable_prompt_truncation: bool = True
    # repetition penalty for response generation
    repetition_penalty: float = 1.0

    # lora config
    lora_configs: Optional[List[LoRAConfig]] = None
    fully_sharded_loras: bool = False
    max_cpu_loras: Optional[int] = None

    # rope config
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None

    # tinker config
    tinker: TinkerConfig = field(default_factory=TinkerConfig)


@dataclass
class InferenceModelConfig:
    # ! DO NOT SET in explorer.rollout_model, automatically set from config.model.model_path
    model_path: Optional[str] = None

    engine_type: str = "vllm"
    engine_num: int = 1
    tensor_parallel_size: int = 1
    use_v1: bool = True
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    seed: int = 42

    # rollout args, ! DO NOT SET
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    logprobs: Optional[int] = None

    # if not set, use `model.max_model_len`
    max_model_len: Optional[int] = None
    # if not set, use `model.max_prompt_tokens`
    max_prompt_tokens: Optional[int] = None
    # if not set, use `model.max_response_tokens`
    max_response_tokens: Optional[int] = None
    # if not set, use `model.min_response_tokens`
    min_response_tokens: Optional[int] = None
    # if not set, use `model.enable_prompt_truncation`
    enable_prompt_truncation: Optional[bool] = None
    # If not set, use `model.repetition_penalty`
    repetition_penalty: Optional[float] = None
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
    enable_log_requests: bool = False  # whether to enable request logging in vLLM API server

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

    # ! DO NOT SET, rope config
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None


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

    # aggregation mode for losses: 'token-mean' or 'seq-mean-token-sum' or 'seq-mean-token-mean' or 'seq-mean-token-sum-norm'
    # If not set, use 'token-mean'
    loss_agg_mode: Optional[str] = None


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

    taskset: Optional[TasksetConfig] = None
    tasksets: List[TasksetConfig] = field(default_factory=list)
    eval_tasksets: List[TasksetConfig] = field(default_factory=list)
    # The following args provide default values for the corresponding args in `taskset` and `eval_tasksets`
    default_workflow_type: Optional[str] = None
    default_eval_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None


@dataclass
class TrainerInput:
    """Config for trainer input."""

    # The main experience buffer to be used in trainer
    # Commonly, it is also the output buffer of the Explorer
    experience_buffer: Optional[ExperienceBufferConfig] = None

    # Some auxiliary buffers to facilitate training (e.g., data mixing)
    auxiliary_buffers: Dict[str, ExperienceBufferConfig] = field(default_factory=dict)


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
    max_timeout: int = 1800  # wait each task for 30 minutes at most
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

    # for serve mode proxy
    proxy_port: int = 8010
    # listen on all interfaces by default
    listen_address: str = "0.0.0.0"
    # check the running status of the server every 60 seconds
    service_status_check_interval: int = 60
    # keep at least 1 model in running status
    min_running_model_num: int = 1
    # db url for proxy history recorder, if not set, use proxy_history.db in buffer cache dir
    db_url: Optional[str] = None

    # Experimental feature
    over_rollout: OverRolloutConfig = field(default_factory=OverRolloutConfig)
    dynamic_timeout: DynamicTimeoutConfig = field(default_factory=DynamicTimeoutConfig)
    # report runner state every `runner_state_report_interval` seconds, 0 to disable
    runner_state_report_interval: int = 0


@dataclass
class TrainerConfig:
    name: str = TRAINER_NAME
    trainer_type: str = "verl"
    trainer_strategy: str = "fsdp"  # "fsdp", "fsdp2" or "megatron"
    save_interval: int = 0
    enable_preview: bool = True  # enable rollout preview in wandb
    total_steps: Optional[
        int
    ] = None  # total training steps, training stops when reaching this step, None means no limit

    save_hf_checkpoint: str = "last"  # whether to save checkpoint in HuggingFace format
    # "always": save all checkpoints in HF format
    # "never": never save checkpoint in HF format
    # "last": only save the last checkpoint in HF format

    # trainer configs
    grad_clip: float = 1.0
    use_dynamic_bsz: bool = True
    # if None, automatically set to ceil(2 * model.max_model_len / ulysses_sequence_parallel_size)
    max_token_len_per_gpu: Optional[int] = None
    ulysses_sequence_parallel_size: int = 1  # sp size
    fix_actor_microbatch_loss_scale: bool = False  # EXPERIMENTAL
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

    def _check_explorer_input(self) -> None:
        from trinity.buffer.selector import SELECTORS

        if self.mode in {"train", "serve"}:
            # no need to check explorer_input in serve mode
            return

        explorer_input = self.buffer.explorer_input

        if explorer_input.taskset:
            if len(explorer_input.tasksets) > 0:
                raise ValueError("Do not support setting `taskset` and `tasksets` simultaneously!")
            explorer_input.tasksets = [explorer_input.taskset]
            explorer_input.taskset = None
        elif self.mode != "bench" and len(explorer_input.tasksets) == 0:
            raise ValueError("At least one taskset should be provided in explorer_input!")

        for i, taskset in enumerate(explorer_input.tasksets):
            if not taskset.path:
                raise ValueError(
                    "`buffer.explorer_input.taskset.path` is required, please set it to the path of the taskset."
                )
            if not taskset.name:
                taskset.name = f"taskset_{i}"
            if taskset.repeat_times is None or taskset.repeat_times != self.algorithm.repeat_times:
                taskset.repeat_times = self.algorithm.repeat_times
                logger.info(
                    "`buffer.explorer_input.taskset.repeat_times` is set to `algorithm.repeat_times`"
                    f" (={self.algorithm.repeat_times})."
                )
            taskset.total_epochs = self.buffer.total_epochs
            taskset.total_steps = self.buffer.total_steps
            taskset.batch_size = self.buffer.batch_size
            set_if_none(taskset, "default_workflow_type", explorer_input.default_workflow_type)
            set_if_none(taskset, "default_reward_fn_type", explorer_input.default_reward_fn_type)
            set_if_none(taskset, "ray_namespace", self.ray_namespace)
            set_if_none(taskset.rollout_args, "temperature", self.model.temperature)
            set_if_none(taskset.rollout_args, "top_p", self.model.top_p)
            set_if_none(taskset.rollout_args, "top_k", self.model.top_k)
            set_if_none(taskset.rollout_args, "logprobs", self.model.logprobs)
            set_if_none(taskset.rollout_args, "max_tokens", self.model.max_response_tokens)
            set_if_none(taskset.format, "chat_template", self.model.custom_chat_template)

            # check if selector is supported
            selector = SELECTORS.get(taskset.task_selector.selector_type)
            if selector is None:
                raise ValueError(
                    f"Selector {taskset.task_selector.selector_type} is not supported."
                )

        for idx, dataset in enumerate(explorer_input.eval_tasksets):
            if not dataset.path:
                raise ValueError(f"Eval dataset [{dataset}]'s path is not configured.")
            dataset.is_eval = True
            dataset.batch_size = self.buffer.batch_size
            if not dataset.name:
                dataset.name = f"eval_taskset_{idx}"

            # eval_workflow has higher priority than workflow in eval tasksets, so we set it first
            set_if_none(dataset, "default_workflow_type", explorer_input.default_eval_workflow_type)
            set_if_none(dataset, "default_workflow_type", explorer_input.default_workflow_type)
            set_if_none(dataset, "default_reward_fn_type", explorer_input.default_reward_fn_type)
            set_if_none(dataset, "ray_namespace", self.ray_namespace)
            set_if_none(dataset.rollout_args, "temperature", self.model.temperature)
            set_if_none(dataset.rollout_args, "top_p", self.model.top_p)
            set_if_none(dataset.rollout_args, "top_k", self.model.top_k)
            set_if_none(dataset.rollout_args, "logprobs", self.model.logprobs)
            set_if_none(dataset.rollout_args, "max_tokens", self.model.max_response_tokens)

    def _check_trainer_input(self) -> None:
        if self.mode == "bench":
            # no need to check trainer_input in bench mode
            return

        trainer_input = self.buffer.trainer_input
        experience_buffer = trainer_input.experience_buffer

        if experience_buffer is None:
            experience_buffer = trainer_input.experience_buffer = ExperienceBufferConfig(
                name="experience_buffer",
                storage_type=StorageType.QUEUE.value,
            )
            logger.info(f"Auto set `buffer.trainer_input.experience_buffer` to {experience_buffer}")
        elif experience_buffer.storage_type == StorageType.FILE.value and self.mode == "both":
            logger.warning(
                "`FILE` storage is not supported to use as experience_buffer in `both` mode, use `QUEUE` instead."
            )
            experience_buffer.storage_type = StorageType.QUEUE.value

        if not experience_buffer.name:
            experience_buffer.name = "experience_buffer"

        if not experience_buffer.path:
            experience_buffer.path = self._default_storage_path(
                experience_buffer.storage_type, experience_buffer.name
            )
            logger.warning(
                f"Auto set `buffer.trainer_input.experience_buffer.path` to {experience_buffer.path}"
            )

        from trinity.algorithm import ALGORITHM_TYPE

        experience_buffer.schema_type = ALGORITHM_TYPE.get(self.algorithm.algorithm_type).schema
        experience_buffer.batch_size = self.buffer.train_batch_size
        experience_buffer.tokenizer_path = self.model.model_path
        set_if_none(experience_buffer, "ray_namespace", self.ray_namespace)
        set_if_none(experience_buffer.format, "chat_template", self.model.custom_chat_template)
        for aux_name, aux_buffer in trainer_input.auxiliary_buffers.items():
            aux_buffer.batch_size = self.buffer.train_batch_size
            aux_buffer.tokenizer_path = self.model.model_path
            set_if_none(aux_buffer, "ray_namespace", self.ray_namespace)
            if aux_buffer.path is None or aux_buffer.path == "":
                raise ValueError(
                    f"`buffer.trainer_input.auxiliary_buffers[{aux_name}].path` is required, "
                    f"please set it to the path of the auxiliary buffer."
                )

        if self.mode == "train":
            assert (
                experience_buffer is not None
            ), "`buffer.trainer_input.experience_buffer` is required when `mode` is `train`."
            experience_buffer.total_epochs = self.buffer.total_epochs
            experience_buffer.total_steps = self.buffer.total_steps

    def _default_storage_path(self, storage_type: str, name: str) -> str:
        if storage_type == StorageType.SQL.value:
            return "sqlite:///" + os.path.join(self.buffer.cache_dir, f"{name}.db")  # type: ignore[arg-type]
        else:
            return os.path.join(self.buffer.cache_dir, f"{name}.jsonl")  # type: ignore[arg-type]

    def _check_data_processor(self) -> None:
        # check input/output buffers in pipelines
        experience_pipeline = self.data_processor.experience_pipeline
        if experience_pipeline is not None and self.mode in {"explore", "both", "serve"}:
            if experience_pipeline.save_input and experience_pipeline.input_save_path is None:
                experience_pipeline.input_save_path = self._default_storage_path(
                    StorageType.SQL.value, "explorer_output"
                )
                logger.info(
                    f"Auto set `data_processor.experience_pipeline.input_save_path` to {experience_pipeline.input_save_path}"
                )

        task_pipeline = self.data_processor.task_pipeline
        if task_pipeline is not None and self.mode in {"explore", "train", "both"}:
            if task_pipeline.output is None:
                if self.mode != "train":
                    if len(self.buffer.explorer_input.tasksets) > 0:
                        task_pipeline.output = self.buffer.explorer_input.tasksets[0]
                    else:
                        raise ValueError(
                            "At least one taskset should be provided in explorer_input!"
                        )
                elif self.mode == "train" and self.algorithm.algorithm_type in {"dpo", "sft"}:
                    task_pipeline.output = self.buffer.trainer_input.experience_buffer
                else:
                    raise ValueError(
                        "`data_processor.task_pipeline.output` is missing. Please set it to the desired output storage config."
                    )
            if task_pipeline.output.path and os.path.exists(task_pipeline.output.path):
                raise ValueError(
                    f"Task pipeline output path {task_pipeline.output.path} already exists.\n"
                    "Please choose a different output path to avoid overwriting."
                )

    def _check_buffer(self) -> None:  # noqa: C901
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

        # create buffer.cache_dir at <checkpoint_root_dir>/<project>/<name>/buffer
        self.buffer.cache_dir = os.path.abspath(os.path.join(self.checkpoint_job_dir, "buffer"))
        try:
            os.makedirs(self.buffer.cache_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create buffer dir {self.buffer.cache_dir}, please check "
                f"your checkpoint directory: {self.checkpoint_job_dir}"
            ) from e

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

        self._check_explorer_input()
        self._check_trainer_input()
        self._check_data_processor()

    def _check_algorithm(self) -> None:
        from trinity.algorithm import (
            ADVANTAGE_FN,
            ALGORITHM_TYPE,
            ENTROPY_LOSS_FN,
            KL_FN,
            POLICY_LOSS_FN,
            SAMPLE_STRATEGY,
        )

        algorithm = ALGORITHM_TYPE.get(self.algorithm.algorithm_type)
        algorithm.check_config(self)
        default_config = {
            "sample_strategy": "warmup",
            "policy_loss_fn": "ppo",
            "advantage_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
            "loss_agg_mode": "token-mean",
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
        if "loss_agg_mode" in self.algorithm.policy_loss_fn_args:  # type: ignore [operator]
            # override loss_agg_mode in policy_loss_fn_args
            self.algorithm.policy_loss_fn_args["loss_agg_mode"] = self.algorithm.loss_agg_mode  # type: ignore [index]

    def _check_model(self) -> None:
        model = self.model
        if not model.critic_model_path:
            model.critic_model_path = model.model_path

        if model.tinker.enable:
            self._check_tinker()

        # check template
        if model.chat_template_path is not None and model.custom_chat_template is None:
            try:
                with open(model.chat_template_path, "r") as f:
                    model.custom_chat_template = f.read()
            except Exception as e:
                raise ValueError(
                    f"Failed to read chat template from {model.chat_template_path}: {e}"
                )

        # check max_model_len, max_prompt_tokens, max_response_tokens
        self._check_model_len()

    def _check_tinker(self) -> None:
        model = self.model
        from trinity.algorithm import ALGORITHM_TYPE

        algorithm = ALGORITHM_TYPE.get(self.algorithm.algorithm_type)
        if algorithm.use_critic:
            raise ValueError("Critic model is not supported when using tinker!")

        import tinker

        service_client = tinker.ServiceClient()
        supported_models = {
            item.model_name for item in service_client.get_server_capabilities().supported_models
        }
        if model.model_path not in supported_models:
            logger.error(f"Supported models: {supported_models}")
            raise ValueError(f"{model.model_path} is not supported by tinker!")

        if (
            self.algorithm.entropy_loss_fn != "none"
            and self.algorithm.entropy_loss_fn_args.get("entropy_coef", 0.0) != 0.0
        ):
            logger.warning(
                "The entropy in Tinker trainer is an estimated value; "
                "it is recommended to set `entropy_coef` to 0."
            )

        if self.explorer.rollout_model.engine_type != "tinker":
            self.explorer.rollout_model.engine_type = "tinker"
            logger.warning("Rollout model engine type is set to `tinker`.")

        if self.trainer.trainer_type != "tinker":
            self.trainer.trainer_type = "tinker"
            logger.warning("Trainer type is set to `tinker`.")

        if self.synchronizer.sync_method == SyncMethod.NCCL:
            self.synchronizer.sync_method = SyncMethod.CHECKPOINT
            logger.warning(
                "Tinker do not support NCCL, `synchronizer.sync_method` is set to `checkpoint`."
            )

    def _check_model_len(self) -> None:
        model = self.model
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

        if model.enable_prompt_truncation is True:
            if model.max_prompt_tokens is None:
                raise ValueError(
                    "When `model.enable_prompt_truncation` is True, `model.max_prompt_tokens` must be set properly. This function does not work with OpenAI API mode."
                )
            logger.warning(
                f"`enable_prompt_truncation` is set to True; the prompt will be truncated to `max_prompt_tokens`={model.max_prompt_tokens} tokens if it is too long."
            )
        else:
            logger.warning(
                "`enable_prompt_truncation` is set to False; please make sure the prompt is not too long and `max_model_len` is large enough, otherwise prompt length + response length may exceed `max_model_len`!"
            )

    def _check_explorer(self) -> None:
        rollout_args = ["temperature", "top_p", "top_k", "logprobs", "repetition_penalty"]
        length_args = [
            "max_model_len",
            "max_prompt_tokens",
            "max_response_tokens",
            "min_response_tokens",
            "enable_prompt_truncation",
        ]
        rope_args = ["rope_scaling", "rope_theta"]
        model_args = rollout_args + length_args + rope_args
        set_if_none(self.explorer.rollout_model, "model_path", self.model.model_path)
        for args in model_args:
            set_if_none(self.explorer.rollout_model, args, getattr(self.model, args))
        if (
            self.explorer.rollout_model.chat_template is None
            and self.model.custom_chat_template is not None
        ):
            self.explorer.rollout_model.chat_template = self.model.custom_chat_template
        for aux_model in self.explorer.auxiliary_models:
            if not aux_model.model_path:
                raise ValueError("auxiliary model's model_path is required.")
            for args in model_args:
                set_if_none(aux_model, args, getattr(self.model, args))

        if self.explorer.rollout_model.engine_type != "tinker":
            # check gpu number
            rollout_gpu_num = (
                self.explorer.rollout_model.tensor_parallel_size
                * self.explorer.rollout_model.engine_num
                + sum(
                    (
                        model.tensor_parallel_size * model.engine_num
                        for model in self.explorer.auxiliary_models
                    )
                )
            )
            assert self.cluster.node_num is not None
            assert self.cluster.gpu_per_node is not None
            total_gpu_num = self.cluster.node_num * self.cluster.gpu_per_node
            if self.mode in ["explore", "bench", "serve"] and rollout_gpu_num > total_gpu_num:
                raise ValueError(
                    f"Total GPU number ({total_gpu_num}) is less than the number of GPUs required for rollout ({rollout_gpu_num})."
                )
            elif self.mode == "both" and rollout_gpu_num >= total_gpu_num:
                raise ValueError(
                    f"Not enough GPUs for trainer in 'both' mode. Explorer requires {rollout_gpu_num} GPUs, but total available GPUs are {total_gpu_num}."
                )

        if self.explorer.over_rollout.ratio > 0.0:
            if not (0.0 <= self.explorer.over_rollout.ratio < 1.0):
                raise ValueError("over_rollout_ratio should be in [0.0, 1.0)")
            if self.synchronizer.sync_style == SyncStyle.FIXED:
                raise ValueError(
                    "over_rollout_ratio is not compatible with fixed sync_style, please set "
                    "`synchronizer.sync_style` to `dynamic_by_explorer` or `dynamic_by_trainer`."
                )

        # for lora configs
        if not self.model.tinker.enable and self.model.lora_configs is not None:
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
            # set trainer.save_hf_checkpoint to "last" to make sure next stage can load from HF checkpoint
            new_config.trainer.save_hf_checkpoint = "last"
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
            self._check_explorer()

        # check synchronizer
        self.synchronizer.ray_namespace = self.ray_namespace
        self.synchronizer.explorer_world_size = (
            self.explorer.rollout_model.engine_num
            * self.explorer.rollout_model.tensor_parallel_size
        )
        if self.synchronizer.sync_method == SyncMethod.NCCL:
            if self.mode in ["train", "explore", "bench", "serve"]:
                self.synchronizer.sync_method = SyncMethod.CHECKPOINT
                logger.warning(
                    f"`{self.mode}` mode does not support NCCL synchronization, set `synchronizer.sync_method` to `checkpoint`."
                )
            if self.model.lora_configs is not None:
                self.synchronizer.sync_method = SyncMethod.CHECKPOINT
                logger.warning(
                    "LoRA is not supported with NCCL synchronization, set `synchronizer.sync_method` to `checkpoint`."
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
        if self.mode in ["train", "both", "bench"] or self.trainer.trainer_strategy == "megatron":
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
                if self.trainer.save_hf_checkpoint not in {"last", "always", "never"}:
                    raise ValueError(
                        f"Invalid trainer.save_hf_checkpoint: {self.trainer.save_hf_checkpoint}, "
                        "must be one of 'last', 'always', or 'never'."
                    )
                self.trainer.trainer_config.synchronize_config(self)
            elif self.trainer.trainer_type == "tinker":
                self.trainer.trainer_config = None
            else:
                raise ValueError(f"Invalid trainer type: {self.trainer_type}")

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
