from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from trinity.common.config import BufferConfig, Config, SynchronizerConfig
from trinity.common.constants import AlgorithmType


@dataclass
class Data:
    tokenizer: Optional[str] = None
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    max_prompt_length: int = 512
    max_response_length: int = 512
    train_batch_size: int = 1024
    val_batch_size: Optional[int] = None
    return_raw_input_ids: bool = False
    return_raw_chat: bool = False
    shuffle: bool = True
    filter_overlong_prompts: bool = False
    truncation: str = "error"
    image_key: str = "images"


@dataclass
class ActorModel:
    path: str = ""
    external_lib: Optional[str] = None
    override_config: Dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = False


@dataclass
class Optim:
    lr: float = 1e-6
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: Optional[float] = 0.0
    warmup_style: str = "constant"
    total_training_steps: int = -1
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class WrapPolicy:
    min_num_params: int = 0


@dataclass
class FSDPConfig:
    wrap_policy: WrapPolicy = field(default_factory=WrapPolicy)
    min_num_params: int = 0
    param_offload: bool = False
    optimizer_offload: bool = False
    fsdp_size: int = -1


@dataclass
class Checkpoint:
    contents: List[str] = field(default_factory=lambda: ["model", "hf_model", "optimizer", "extra"])


@dataclass
class Actor:
    strategy: str = "fsdp"
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: int = 1
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = (
        16384  # n * ${data.max_prompt_length} + ${data.max_response_length}
    )
    grad_clip: float = 1.0
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.001
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    ulysses_sequence_parallel_size: int = 1
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    optim: Optim = field(default_factory=Optim)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    algorithm_type: AlgorithmType = AlgorithmType.PPO
    tau: float = 0.001  # strength of regularization w.r.t. old / ref policy
    opmd_baseline: str = "mean"  # mean / logavgexp, applicable to opmd
    use_uid: bool = False  # True / False, applicable to pairwise_opmd


@dataclass
class Ref:
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: int = 1
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 0
    ulysses_sequence_parallel_size: int = 1


@dataclass
class Rollout:
    name: str = "vllm"
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    use_fire_sampling: bool = False
    prompt_length: int = 0
    response_length: int = 0
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    free_cache_engine: bool = True
    load_format: str = "dummy_dtensor"
    tensor_model_parallel_size: int = 2
    max_num_batched_tokens: int = 8192
    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: int = 1
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 0
    disable_log_stats: bool = True
    enable_chunked_prefill: bool = True
    do_sample: bool = True
    n: int = 1  # > 1 for grpo


@dataclass
class ActorRolloutRef:
    hybrid_engine: bool = False
    model: ActorModel = field(default_factory=ActorModel)
    actor: Actor = field(default_factory=Actor)
    ref: Ref = field(default_factory=Ref)
    rollout: Rollout = field(default_factory=Rollout)
    synchronizer: Optional[SynchronizerConfig] = None


@dataclass
class CriticModel:
    path: str = ""
    tokenizer_path: str = ""
    override_config: Dict[str, str] = field(default_factory=dict)
    external_lib: Optional[str] = None
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = False
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class Critic:
    strategy: str = "fsdp"
    optim: Optim = field(default_factory=Optim)
    model: CriticModel = field(default_factory=CriticModel)
    ppo_mini_batch_size: int = 0
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: int = 1
    forward_micro_batch_size: Optional[int] = None
    forward_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 0
    forward_max_token_len_per_gpu: int = 0
    ulysses_sequence_parallel_size: int = 0
    ppo_epochs: int = 0
    shuffle: bool = False
    grad_clip: float = 0.0
    cliprange_value: float = 0.0
    checkpoint: Checkpoint = field(default_factory=Checkpoint)


@dataclass
class _RewardModel:
    input_tokenizer: Optional[str] = None
    path: str = ""
    external_lib: Optional[str] = None
    use_remove_padding: bool = False
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class RewardModel:
    enable: bool = False
    strategy: str = "fsdp"
    model: _RewardModel = field(default_factory=_RewardModel)
    micro_batch_size_per_gpu: int = 1
    max_length: Optional[int] = None
    ulysses_sequence_parallel_size: int = 1
    use_dynamic_bsz: bool = False
    forward_max_token_len_per_gpu: int = 0
    reward_manager: str = "naive"


@dataclass
class CustomRewardFunction:
    path: Optional[str] = None
    name: str = "compute_score"


@dataclass
class KL_Ctrl:
    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: float = 10000
    target_kl: float = 0.1


@dataclass
class Algorithm:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KL_Ctrl = field(default_factory=KL_Ctrl)


@dataclass
class Trainer:
    balance_batch: bool = True
    total_epochs: int = 30
    total_training_steps: Optional[int] = None
    project_name: str = ""
    experiment_name: str = ""
    logger: List[str] = field(default_factory=list)
    val_generations_to_log_to_wandb: int = 0
    nnodes: int = 0
    n_gpus_per_node: int = 0
    save_freq: int = 0
    resume_mode: str = "auto"
    resume_from_path: str = ""
    test_freq: int = 0
    critic_warmup: int = 0
    default_hdfs_dir: Optional[str] = None
    remove_previous_ckpt_in_save: bool = False  # deprecated
    del_local_ckpt_after_load: bool = False
    default_local_dir: str = ""
    val_before_train: bool = False
    training_rollout_mode: str = "parallel"
    enable_exp_buffer: bool = True
    sync_freq: int = 0
    sft_warmup_iteration: int = 0
    max_actor_ckpt_to_keep: Optional[int] = None
    max_critic_ckpt_to_keep: Optional[int] = None


@dataclass
class veRLConfig:
    data: Data = field(default_factory=Data)
    actor_rollout_ref: ActorRolloutRef = field(default_factory=ActorRolloutRef)
    critic: Critic = field(default_factory=Critic)
    reward_model: RewardModel = field(default_factory=RewardModel)
    custom_reward_function: CustomRewardFunction = field(default_factory=CustomRewardFunction)
    algorithm: Algorithm = field(default_factory=Algorithm)
    trainer: Trainer = field(default_factory=Trainer)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    synchronizer: Optional[SynchronizerConfig] = None
    enable_preview: bool = True

    def synchronize_config(self, config: Config) -> None:
        """Synchronize config."""
        rollout_gpu_num = config.explorer.tensor_parallel_size * config.explorer.engine_num
        rollout_node_num = rollout_gpu_num // config.cluster.gpu_per_node
        self.trainer.nnodes = config.cluster.node_num - rollout_node_num
        self.actor_rollout_ref.model.path = config.model.model_path
        self.critic.model.path = config.model.critic_model_path
        self.critic.model.tokenizer_path = config.model.critic_model_path

        if config.cluster.node_num == 1:
            # for single node scenarios, rollout and training are on the same node
            self.trainer.n_gpus_per_node = config.cluster.gpu_per_node - rollout_gpu_num
        else:
            # for multi-node scenarios, some nodes for rollout, others for training
            self.trainer.n_gpus_per_node = config.cluster.gpu_per_node
        self.trainer.sync_freq = config.synchronizer.sync_iteration_interval
        self.trainer.save_freq = config.trainer.save_interval
        self.synchronizer = config.synchronizer
        self.actor_rollout_ref.synchronizer = config.synchronizer
        self.buffer = config.buffer
        world_size = self.trainer.nnodes * self.trainer.n_gpus_per_node
        if config.data.batch_size % world_size != 0:
            raise ValueError(
                f"batch_size ({config.data.batch_size}) must be divisible by ({world_size})"
            )
        # TODO: use dynamic read_batch_size to support multi-round scenarios
        # Get the experiences of one explore iteration
        self.buffer.pad_token_id = config.buffer.pad_token_id
        self.trainer.project_name = config.monitor.project
        self.trainer.experiment_name = config.monitor.name
        self.data.train_batch_size = self.buffer.read_batch_size
        self.trainer.default_local_dir = config.model.checkpoint_path
        self.trainer.sft_warmup_iteration = config.trainer.sft_warmup_iteration
        self.actor_rollout_ref.actor.ppo_mini_batch_size = config.data.batch_size
        self.actor_rollout_ref.rollout.temperature = config.explorer.temperature
        self.actor_rollout_ref.rollout.n = config.explorer.repeat_times
        batch_size_per_gpu = self.buffer.read_batch_size // world_size
        self.actor_rollout_ref.actor.algorithm_type = config.trainer.algorithm_type

        if self.actor_rollout_ref.actor.algorithm_type.is_dpo():  # for DPO
            print("Warning: DPO micro batch size is doubled for computing loss.")
            self.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu *= 2  # type: ignore
            self.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu *= 2  # type: ignore
        if batch_size_per_gpu % self.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu != 0:  # type: ignore
            raise ValueError(
                f"batch_size_per_gpu ({batch_size_per_gpu}) must be divisible by "
                f"actor.ppo_micro_batch_size_per_gpu ({self.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu})"
            )
        if batch_size_per_gpu % self.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu != 0:  # type: ignore
            raise ValueError(
                f"batch_size_per_gpu ({batch_size_per_gpu}) must be divisible by "
                f"ref.log_prob_micro_batch_size_per_gpu ({self.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu})"
            )
        if batch_size_per_gpu % self.critic.ppo_micro_batch_size_per_gpu != 0:  # type: ignore
            raise ValueError(
                f"batch_size_per_gpu ({batch_size_per_gpu}) must be divisible by "
                f"critic.ppo_micro_batch_size_per_gpu ({self.critic.ppo_micro_batch_size_per_gpu})"
            )
        # TODO: check other fields
        self.enable_preview = config.trainer.enable_preview


def load_config(config_path: str) -> veRLConfig:
    schema = OmegaConf.structured(veRLConfig)
    yaml_config = OmegaConf.load(config_path)
    try:
        config = OmegaConf.merge(schema, yaml_config)
        return OmegaConf.to_object(config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
