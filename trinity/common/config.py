# -*- coding: utf-8 -*-
"""Configs for RFT."""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from trinity.common.constants import (
    EXPLORER_NAME,
    TRAINER_NAME,
    OpType,
    PromptType,
    ReadStrategy,
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
    chat_template: str = ""  # deprecated

    system_prompt: Optional[str] = None
    reply_prefix: Optional[str] = None

    # for sample-level task controlling
    reward_fn_key: str = ""
    workflow_key: str = ""
    # for math dataset
    solution_key: str = "solution"

    # for reward dataset
    reward_key: str = "reward"

    # for dpo dataset
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"

    # for unpaired preference dataset
    label_key: str = ""


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    logprobs: int = 0  # vLLM return `logprobs + 1` elements
    # repeat each task for `n` times (for GPRO-like algorithms)
    # this field will be automatically set to `algorithm.repeat_times` in
    # `buffer.explorer_input.taskset.rollout_args`
    # ! DO NOT SET in `buffer.explorer_input.taskset.rollout_args`
    n: int = 1


@dataclass
class StorageConfig:
    """Storage config."""

    name: str = ""
    storage_type: StorageType = StorageType.FILE
    path: Optional[str] = None

    # only available for StorageType.FILE. When requiring data processing on raw data, set the raw to True.
    raw: bool = False

    # used for StorageType.FILE
    split: str = "train"
    subset_name: Optional[str] = None
    format: FormatConfig = field(default_factory=FormatConfig)
    index: int = 0

    # used for StorageType.SQL/FILE
    wrap_in_ray: bool = True

    # used for StorageType.QUEUE
    capacity: int = 10000

    # used for rollout tasks
    default_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None
    rollout_args: GenerationConfig = field(default_factory=GenerationConfig)
    workflow_args: dict = field(default_factory=dict)

    # get storage from existing experiment
    ray_namespace: Optional[str] = None

    # ! DO NOT SET, automatically set from algorithm.algorithm_type
    algorithm_type: Optional[str] = None

    # ! DO NOT SET, automatically set from buffer.total_epochs
    total_epochs: int = 1  # automatically set

    # ! DO NOT SET,  automatically set corresponding to train/eval
    task_type: TaskType = TaskType.EXPLORE


@dataclass
class RewardShapingConfig:
    """Config for reward shaping."""

    stats_key: str = ""
    op_type: OpType = OpType.ADD
    weight: float = 1.0


@dataclass
class DataPipelineConfig:
    """Config for data pipeline."""

    # I/O buffer
    input_buffers: List[StorageConfig] = field(default_factory=list)
    output_buffer: StorageConfig = field(default_factory=StorageConfig)

    # data format
    format: FormatConfig = field(default_factory=FormatConfig)

    # data active iterator related
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

    # reward shaping related, only available for experience pipeline
    reward_shaping: Optional[List[RewardShapingConfig]] = field(default_factory=list)


@dataclass
class DataProcessorConfig:
    """Data-Juicer config"""

    data_processor_url: Optional[str] = None

    # support two types of data pipelines for now
    # 1. For task. Data preprocessing from raw dataset to the task set
    task_pipeline: Optional[DataPipelineConfig] = None
    # 2. For experience. Data processing for rollouts
    experience_pipeline: Optional[DataPipelineConfig] = None


@dataclass
class ModelConfig:
    # source model path
    model_path: str = ""
    critic_model_path: str = ""
    max_prompt_tokens: Optional[int] = None
    max_response_tokens: Optional[int] = None


@dataclass
class InferenceModelConfig:
    # ! DO NOT SET in explorer.rollout_model, automatically set from config.model.model_path
    model_path: str = ""

    # support `vllm` or `vllm_async`,
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

    # if not set, use `model.max_prompt_tokens`
    max_prompt_tokens: Optional[int] = None
    # if not set, use `model.max_response_tokens`
    max_response_tokens: Optional[int] = None

    # override chat template in model
    chat_template: Optional[str] = None

    # For Qwen3
    enable_thinking: bool = False

    # For OpenAI API
    enable_openai_api: bool = False

    # ! DO NOT SET
    bundle_indices: str = ""


@dataclass
class AlgorithmConfig:
    """Config for algorithm."""

    algorithm_type: str = "ppo"
    # for GRPO-like algorithms, repeat each task for `repeat_times` times
    repeat_times: int = 1

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

    # used for SFT warmup
    # TODO: move this to SFT warmup
    use_token_level_loss: bool = True


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
    # The following args provide default values for the corresponding args in `taskset` and `eval_tasksets`
    default_workflow_type: Optional[str] = None
    default_reward_fn_type: Optional[str] = None
    system_prompt: Optional[str] = None
    reply_prefix: Optional[str] = None


@dataclass
class TrainerInput:
    """Config for trainer input."""

    experience_buffer: Optional[StorageConfig] = None
    sft_warmup_dataset: Optional[StorageConfig] = None
    read_experience_strategy: Optional[ReadStrategy] = None
    sft_warmup_steps: int = 0


@dataclass
class BufferConfig:
    """Config for buffer."""

    batch_size: int = 1
    total_epochs: int = 1

    # for explorer
    explorer_input: ExplorerInput = field(default_factory=ExplorerInput)
    explorer_output: Optional[StorageConfig] = None  # currently do not set

    # for trainer
    trainer_input: TrainerInput = field(default_factory=TrainerInput)

    # for storage connection
    max_retry_times: int = 3
    max_retry_interval: int = 1

    # ! DO NOT SET FOLLOWING FIELDS
    read_batch_size: int = 1  # automatically set
    tokenizer_path: Optional[str] = None  # automatically set
    pad_token_id: Optional[int] = None  # automatically set
    cache_dir: Optional[str] = None  # automatically set


@dataclass
class ExplorerConfig:
    """Config for explorer."""

    name: str = EXPLORER_NAME
    # for workflow runner
    # number of workflow runners.
    # For sync engine (vllm), it should be equal to `engine_num`.
    # For async engine (vllm_async), it can be larger than `engine_num`, e.g. 16 * `engine_num`
    runner_num: int = 1
    max_timeout: int = 900  # wait each task for 15 minutes
    max_retry_times: int = 2  # retry each task for 2 times if it fails or timeout
    env_vars: dict = field(default_factory=dict)

    # for inference models
    # for rollout model
    rollout_model: InferenceModelConfig = field(default_factory=InferenceModelConfig)
    # for other models used in the custom workflows
    auxiliary_models: List[InferenceModelConfig] = field(default_factory=list)

    # for evaluation
    eval_interval: int = 100
    eval_on_latest_checkpoint: bool = False


@dataclass
class TrainerConfig:
    name: str = TRAINER_NAME
    trainer_type: str = "verl"
    save_interval: int = 0
    enable_preview: bool = True  # enable rollout preview in wandb

    # trainer configs
    actor_grad_clip: Optional[float] = None
    # TODO: extract more train-related params from underlying trainer engine

    # Only one needs to be set for `trainer_config` and `trainer_config_path`
    trainer_config: Any = field(default_factory=dict)
    trainer_config_path: str = ""


@dataclass
class MonitorConfig:
    # TODO: support multiple monitors (List[str])
    monitor_type: str = "tensorboard"
    # the default args for monitor
    monitor_args: Dict = field(default_factory=dict)
    # whether to enable ray timeline profile
    # the output file will be saved to `cache_dir/timeline.json`
    enable_ray_timeline: bool = False
    # ! DO NOT SET, automatically generated as checkpoint_job_dir/monitor
    cache_dir: str = ""


@dataclass
class SynchronizerConfig:
    """Configs for model weight synchronization"""

    # TODO: rename to "checkpoint", "nccl", "ipc"
    sync_method: SyncMethod = SyncMethod.NCCL
    # sync weights every `sync_interval` steps
    sync_interval: int = 1
    # allow explorer to run `sync_offset` steps before sync
    sync_offset: int = 0
    # waiting for `sync_timeout` seconds before timeout in `nccl` method
    sync_timeout: int = 1800
    # wait for the lastest checkpoint to be ready  # TODO: to be used
    wait_for_checkpoint: bool = False

    # ! DO NOT SET, automatically calculated
    explorer_world_size: Optional[int] = None


@dataclass
class Config:
    """Global Configuration"""

    mode: str = "both"  # `explore`, `train`, `both` or `bench`
    project: str = "Trinity-RFT"
    name: str = "rft"
    # the root dir for checkpoints
    checkpoint_root_dir: str = ""
    # ! DO NOT SET, automatically generated as `checkpoint_root_dir/project/name`
    checkpoint_job_dir: str = ""
    # If not set, automatically generated as f"{config.project}-{config.name}"
    ray_namespace: str = ""

    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    data_processor: DataProcessorConfig = field(default_factory=DataProcessorConfig)
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
        pass

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

            # check save_interval
            if self.synchronizer.sync_method == SyncMethod.CHECKPOINT:
                if self.trainer.save_interval != self.synchronizer.sync_interval:
                    logger.warning(
                        f"When `algorithm.algorithm_type` != `dpo` and `synchronizer.sync_method` == `checkpoint`, "
                        f"`trainer.save_interval` will be set to "
                        f"`synchronizer.sync_interval = {self.synchronizer.sync_interval}`."
                    )
                self.trainer.save_interval = self.synchronizer.sync_interval

    def _check_buffer(self) -> None:  # noqa: C901
        # TODO: split this function into different buffer read/writer
        # check explorer_input
        if self.mode != "train" and not self.buffer.explorer_input.taskset.path:
            raise ValueError(
                "`buffer.explorer_input.taskset.path` is required, please set it to the path of the taskset."
            )
        if not self.buffer.explorer_input.taskset.name:
            self.buffer.explorer_input.taskset.name = "taskset"
        self.buffer.explorer_input.taskset.rollout_args.n = self.algorithm.repeat_times
        logger.info(
            "`buffer.explorer_input.taskset.rollout_args.n` is set to `algorithm.repeat_times`"
            f" (={self.algorithm.repeat_times})."
        )
        self.buffer.explorer_input.taskset.task_type = TaskType.EXPLORE
        self.buffer.explorer_input.taskset.total_epochs = self.buffer.total_epochs
        if self.buffer.explorer_input.taskset.default_workflow_type is None:
            self.buffer.explorer_input.taskset.default_workflow_type = (
                self.buffer.explorer_input.default_workflow_type
            )
        if self.buffer.explorer_input.taskset.default_reward_fn_type is None:
            self.buffer.explorer_input.taskset.default_reward_fn_type = (
                self.buffer.explorer_input.default_reward_fn_type
            )
        if self.buffer.explorer_input.taskset.format.system_prompt is None:
            self.buffer.explorer_input.taskset.format.system_prompt = (
                self.buffer.explorer_input.system_prompt
            )
        if self.buffer.explorer_input.taskset.format.reply_prefix is None:
            self.buffer.explorer_input.taskset.format.reply_prefix = (
                self.buffer.explorer_input.reply_prefix
            )
        if self.buffer.explorer_input.taskset.ray_namespace is None:
            self.buffer.explorer_input.taskset.ray_namespace = self.ray_namespace

        remained_tasksets = []
        for idx, dataset in enumerate(self.buffer.explorer_input.eval_tasksets):
            if not dataset.path:
                logger.warning(f"Eval dataset [{dataset}]'s path is not configured. Skip.")
                continue
            dataset.task_type = TaskType.EVAL
            if not dataset.name:
                dataset.name = f"eval_taskset_{idx}"
            if dataset.default_workflow_type is None:
                dataset.default_workflow_type = self.buffer.explorer_input.default_workflow_type
            if dataset.default_reward_fn_type is None:
                dataset.default_reward_fn_type = self.buffer.explorer_input.default_reward_fn_type
            if dataset.format.system_prompt is None:
                dataset.format.system_prompt = self.buffer.explorer_input.system_prompt
            if dataset.format.reply_prefix is None:
                dataset.format.reply_prefix = self.buffer.explorer_input.reply_prefix
            if dataset.ray_namespace is None:
                dataset.ray_namespace = self.ray_namespace
            remained_tasksets.append(dataset)
        self.buffer.explorer_input.eval_tasksets = remained_tasksets

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
            elif self.buffer.trainer_input.experience_buffer.storage_type is StorageType.FILE:
                logger.warning(
                    "`FILE` storage is not supported to use as experience_buffer in `both` mode, use `QUEUE` instead."
                )
                self.buffer.trainer_input.experience_buffer.storage_type = StorageType.QUEUE
        elif self.mode == "train":  # TODO: to be check
            pass
        if self.buffer.trainer_input.experience_buffer is not None:
            self.buffer.trainer_input.experience_buffer.algorithm_type = (
                self.algorithm.algorithm_type
            )
            if self.buffer.trainer_input.experience_buffer.ray_namespace is None:
                self.buffer.trainer_input.experience_buffer.ray_namespace = self.ray_namespace

        # set buffer.explorer_output
        if self.buffer.explorer_output is None:
            self.buffer.explorer_output = self.buffer.trainer_input.experience_buffer
        else:
            self.buffer.explorer_output.algorithm_type = self.algorithm.algorithm_type
            if self.buffer.explorer_output.ray_namespace is None:
                self.buffer.explorer_output.ray_namespace = self.ray_namespace

        # check trainer_input.sft_warmup_dataset
        if (
            self.buffer.trainer_input.sft_warmup_steps > 0
            and self.buffer.trainer_input.sft_warmup_dataset is None
        ):
            raise ValueError(
                "`buffer.trainer_input.sft_warmup_dataset` is required when `buffer.trainer_input.sft_warmup_steps` > 0"
            )
        if self.buffer.trainer_input.sft_warmup_dataset is not None:
            self.buffer.trainer_input.sft_warmup_dataset.algorithm_type = "sft"  # TODO
            if self.buffer.trainer_input.sft_warmup_dataset.ray_namespace is None:
                self.buffer.trainer_input.sft_warmup_dataset.ray_namespace = self.ray_namespace

        # check input/output buffers in experience pipelines
        if self.data_processor.experience_pipeline is not None:
            # collect existing buffers for trinity
            input_buffers = {}
            output_buffers = {}
            # - taskset
            if self.buffer.explorer_input.taskset.name:
                input_buffers[
                    self.buffer.explorer_input.taskset.name
                ] = self.buffer.explorer_input.taskset
            # - explorer output
            if self.buffer.explorer_output and self.buffer.explorer_output.name:
                output_buffers[self.buffer.explorer_output.name] = self.buffer.explorer_output
            # - trainer input: experience buffer
            if (
                self.buffer.trainer_input.experience_buffer
                and self.buffer.trainer_input.experience_buffer.name
            ):
                input_buffers[
                    self.buffer.trainer_input.experience_buffer.name
                ] = self.buffer.trainer_input.experience_buffer
            # - trainer input: sft warmup dataset
            if (
                self.buffer.trainer_input.sft_warmup_dataset
                and self.buffer.trainer_input.sft_warmup_dataset.name
            ):
                input_buffers[
                    self.buffer.trainer_input.sft_warmup_dataset.name
                ] = self.buffer.trainer_input.sft_warmup_dataset

            # when experience pipeline is on, the explorer output and the
            # experience buffer of trainer input should be different
            if self.buffer.explorer_output == self.buffer.trainer_input.experience_buffer:
                raise ValueError(
                    "The explorer output buffer should be different from the experience buffer of the trainer input "
                    "when experience pipeline is provided."
                )

            # NOTICE: For now, input/output buffers for data processors should come from output/input buffers of trinity
            # the input buffers in experience pipeline should come from the output buffers of trinity
            exp_pipeline_input_buffers = self.data_processor.experience_pipeline.input_buffers
            synced_input_buffers = []
            for input_buffer in exp_pipeline_input_buffers:
                if input_buffer.name not in output_buffers:
                    raise ValueError(
                        f"The input buffer {input_buffer.name} of experience pipeline is not found in any output "
                        f"buffers of trinity."
                    )
                synced_input_buffers.append(output_buffers[input_buffer.name])
            self.data_processor.experience_pipeline.input_buffers = synced_input_buffers
            # the output buffers of trinity should come from the input buffers of trinity
            exp_pipeline_output_buffers = self.data_processor.experience_pipeline.output_buffer
            if exp_pipeline_output_buffers.name not in input_buffers:
                raise ValueError(
                    f"The output buffer {exp_pipeline_output_buffers.name} of experience pipeline is not found in any "
                    f"input buffers of trinity."
                )
            else:
                self.data_processor.experience_pipeline.output_buffer = input_buffers[
                    exp_pipeline_output_buffers.name
                ]

        # set read_batch_size / pad_token_id / tokenizer_path
        self.buffer.read_batch_size = self.buffer.batch_size * self.algorithm.repeat_times
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
        # create buffer.cache_dir at <checkpoint_root_dir>/<project>/<name>/buffer
        self.buffer.cache_dir = os.path.abspath(os.path.join(self.checkpoint_job_dir, "buffer"))
        try:
            os.makedirs(self.buffer.cache_dir, exist_ok=True)
        except Exception:
            logger.warning(
                f"Failed to create buffer dir {self.buffer.cache_dir}, please check "
                f"your checkpoint directory: {self.checkpoint_job_dir}"
            )

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
            if getattr(self.algorithm, key, None) is None:
                setattr(self.algorithm, key, value)

        # TODO: simplify the following code
        sample_strategy_cls = SAMPLE_STRATEGY.get(self.algorithm.sample_strategy)
        if sample_strategy_cls is None:
            raise ValueError(f"Invalid sample_strategy: {self.algorithm.sample_strategy}")
        if self.algorithm.sample_strategy_args is None:
            self.algorithm.sample_strategy_args = sample_strategy_cls.default_args()

        policy_fn_cls = POLICY_LOSS_FN.get(self.algorithm.policy_loss_fn)
        if policy_fn_cls is None:
            raise ValueError(f"Invalid policy_loss_fn: {self.algorithm.policy_loss_fn}")
        if self.algorithm.policy_loss_fn_args is None:
            self.algorithm.policy_loss_fn_args = policy_fn_cls.default_args()

        advantage_fn_cls = ADVANTAGE_FN.get(self.algorithm.advantage_fn)
        if advantage_fn_cls is None:
            raise ValueError(f"Invalid advantage_fn: {self.algorithm.advantage_fn}")
        if self.algorithm.advantage_fn_args is None:
            self.algorithm.advantage_fn_args = advantage_fn_cls.default_args()

        kl_loss_fn_cls = KL_FN.get(self.algorithm.kl_loss_fn)
        if kl_loss_fn_cls is None:
            raise ValueError(f"Invalid kl_loss_fn: {self.algorithm.kl_loss_fn}")
        if self.algorithm.kl_loss_fn_args is None:
            self.algorithm.kl_loss_fn_args = kl_loss_fn_cls.default_args()

        kl_penalty_fn_cls = KL_FN.get(self.algorithm.kl_penalty_fn)
        if kl_penalty_fn_cls is None:
            raise ValueError(f"Invalid kl_penalty_fn: {self.algorithm.kl_penalty_fn}")
        if self.algorithm.kl_penalty_fn_args is None:
            self.algorithm.kl_penalty_fn_args = kl_penalty_fn_cls.default_args()

        entropy_loss_fn_cls = ENTROPY_LOSS_FN.get(self.algorithm.entropy_loss_fn)
        if entropy_loss_fn_cls is None:
            raise ValueError(f"Invalid entropy_loss_fn: {self.algorithm.entropy_loss_fn}")
        if self.algorithm.entropy_loss_fn_args is None:
            self.algorithm.entropy_loss_fn_args = entropy_loss_fn_cls.default_args()

    def check_and_update(self) -> None:  # noqa: C901
        """Check and update the config."""
        self._check_deprecated()

        # set namespace
        if self.ray_namespace is None or len(self.ray_namespace) == 0:
            self.ray_namespace = f"{self.project}/{self.name}"

        # check algorithm
        self._check_algorithm()

        # check mode
        if self.mode not in ["explore", "train", "both", "bench"]:
            raise ValueError(f"Invalid mode: {self.mode}")

        # prepare for the checkpoint directory
        if not os.path.isabs(self.checkpoint_root_dir):
            self.checkpoint_root_dir = os.path.join(os.getcwd(), self.checkpoint_root_dir)
        # create a job dir at checkpoint_root_dir/project/name
        self.checkpoint_job_dir = os.path.join(self.checkpoint_root_dir, self.project, self.name)
        os.makedirs(self.checkpoint_job_dir, exist_ok=True)

        # check and update model path
        if self.explorer is not None:
            self.explorer.rollout_model.model_path = self.model.model_path
        if not self.model.critic_model_path:
            self.model.critic_model_path = self.model.model_path

        # check explorer
        if (
            self.explorer.rollout_model.engine_type != "vllm_async"
            and self.explorer.rollout_model.enable_openai_api
        ):
            raise ValueError("OpenAI API server only support `vllm_async` engine.")
        if self.explorer.rollout_model.max_prompt_tokens is None:
            self.explorer.rollout_model.max_prompt_tokens = self.model.max_prompt_tokens
        if self.explorer.rollout_model.max_response_tokens is None:
            self.explorer.rollout_model.max_response_tokens = self.model.max_response_tokens

        # check synchronizer
        self.synchronizer.explorer_world_size = (
            self.explorer.rollout_model.engine_num
            * self.explorer.rollout_model.tensor_parallel_size
        )
        if (
            self.mode in ["train", "explore", "bench"]
            and self.synchronizer.sync_method != SyncMethod.CHECKPOINT
        ):
            self.synchronizer.sync_method = SyncMethod.CHECKPOINT
            logger.warning(
                f"`{self.mode}` mode only supports checkpoint synchronization, set `synchronizer.sync_method` to `checkpoint`."
            )

        self._check_interval()

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
