import math
import os
from abc import ABC, abstractmethod
from datetime import datetime

import ray
from omegaconf import OmegaConf

from trinity.common.config import (
    Config,
    ExperienceBufferConfig,
    TasksetConfig,
    set_if_none,
)
from trinity.common.constants import StorageType, SyncMethod, SyncStyle
from trinity.utils.log import get_logger
from trinity.utils.lora_utils import create_dummy_lora


class ConfigValidator(ABC):
    """Abstract base class for configuration validators.

    Each validator is responsible for checking and potentially modifying specific
    aspects of the global configuration to ensure validity, set defaults, or handle
    deprecated settings.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    @abstractmethod
    def validate(self, config: Config) -> None:
        """Validate and potentially modify the given configuration.

        Args:
            config: The global configuration object to validate and modify.
        """
        pass


class DeprecatedConfigValidator(ConfigValidator):
    """Validator for handling deprecated configuration options.

    Issues warnings when deprecated configuration parameters are used and suggests
    their replacements.
    """

    def validate(self, config: Config) -> None:
        """Check for deprecated configuration options and issue warnings.

        Specifically checks for the deprecated `explorer.runner_num` parameter
        and recommends using `explorer.runner_per_model` instead.

        Args:
            config: The global configuration object to validate.
        """
        if config.explorer.runner_num is not None:
            self.logger.warning(
                "`explorer.runner_num` is deprecated, "
                "please use `explorer.runner_per_model` instead."
            )


class GlobalConfigValidator(ConfigValidator):
    """Validator for global configuration settings.

    Handles validation of the main operating mode, sets up checkpoint directories,
    and configures logging paths. Manages experiment naming conflicts by appending
    timestamps to avoid overwriting existing experiments.
    """

    def validate(self, config: Config) -> None:
        """Validate global configuration settings and set up directory structure.

        - Validates that the mode is one of the supported values
        - Creates absolute checkpoint paths and handles experiment naming conflicts
        - Sets up the log directory path

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If an invalid mode is specified.
        """
        # check mode
        if config.mode not in ["explore", "train", "both", "bench", "serve"]:
            raise ValueError(f"Invalid mode: {config.mode}")

        # prepare for the checkpoint directory
        if not os.path.isabs(config.checkpoint_root_dir):
            config.checkpoint_root_dir = os.path.join(os.getcwd(), config.checkpoint_root_dir)
        # create a job dir at checkpoint_root_dir/project/name
        config.checkpoint_job_dir = os.path.join(
            config.checkpoint_root_dir, config.project, config.group, config.name
        )
        # rename the experiment when necessary
        if not config.continue_from_checkpoint and (
            os.path.exists(config.checkpoint_job_dir) and os.listdir(config.checkpoint_job_dir)
        ):
            if config.mode == "bench":
                self.logger.warning(
                    "For bench mode, `continue_from_checkpoint` is set as `true` "
                    "to enable using existing checkpoints."
                )
                config.continue_from_checkpoint = True
            else:
                ori_name = config.name
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                config.name = f"{ori_name}_{timestamp}"
                config.checkpoint_job_dir = f"{config.checkpoint_job_dir}_{timestamp}"
                self.logger.warning(
                    f"Experiment [{ori_name}] already exists, renamed as {config.name}."
                )
        os.makedirs(config.checkpoint_job_dir, exist_ok=True)

        # check log
        config.log.save_dir = os.path.join(config.checkpoint_job_dir, "log")


class RayClusterConfigValidator(ConfigValidator):
    """Validator for Ray cluster configuration.

    Handles Ray cluster setup including namespace configuration, automatic detection
    of cluster resources (node count and GPUs per node), and GPU allocation validation
    based on the current operating mode and model requirements.
    """

    def validate(self, config: Config) -> None:
        """Validate and configure Ray cluster settings.

        - Sets the Ray namespace if not provided
        - Skips validation if Tinker is enabled
        - Automatically detects cluster information if not provided
        - Validates GPU allocation based on mode and model requirements

        Args:
            config: The global configuration object to validate.

        Raises:
            RuntimeError: If no alive nodes are found in the Ray cluster.
            ValueError: If GPU allocation requirements cannot be satisfied.
        """
        # set namespace
        if config.ray_namespace is None or len(config.ray_namespace) == 0:
            config.ray_namespace = f"{config.project}/{config.name}"

        if config.model.tinker.enable:
            return

        # check cluster infomation
        if not config.cluster.node_num or not config.cluster.gpu_per_node:
            self._set_cluster_info(config)
        self._set_gpu_allocation_info(config)

    def _set_cluster_info(self, config: Config) -> None:
        """Automatically detect and set cluster node and GPU information.

        Initializes Ray if not already initialized, queries the cluster for
        alive nodes and available GPUs, then sets the configuration accordingly.

        Args:
            config: The global configuration object to modify.

        Raises:
            RuntimeError: If no alive nodes are found in the Ray cluster.
        """
        # init ray cluster to detect node_num and gpu_per_node
        was_initialized = ray.is_initialized()
        if not was_initialized:
            ray.init(
                address=config.cluster.ray_address,
                ignore_reinit_error=True,
                namespace=config.ray_namespace,
            )

        alive_nodes = [n for n in ray.nodes() if n["alive"]]
        if not alive_nodes:
            raise RuntimeError("Could not find any alive nodes in the Ray cluster.")

        # set node_num
        if not config.cluster.node_num:
            config.cluster.node_num = len(alive_nodes)
            self.logger.info(f"Auto-detected and set node_num: {config.cluster.node_num}")

        # set gpu_per_node
        if not config.cluster.gpu_per_node:
            gpu_per_node = 0
            for node in alive_nodes:
                node_gpus = node.get("Resources", {}).get("GPU")
                if node_gpus and node_gpus > 0:
                    gpu_per_node = int(node_gpus)
                    break

            config.cluster.gpu_per_node = gpu_per_node
            self.logger.info(f"Auto-detected and set gpu_per_node: {config.cluster.gpu_per_node}")

        if not was_initialized:
            ray.shutdown()

    def _set_gpu_allocation_info(self, config: Config) -> None:
        """Calculate and validate GPU allocation for explorer and trainer components.

        Computes GPU requirements based on model configurations and validates that
        the total available GPUs are sufficient for the requested allocation.

        Args:
            config: The global configuration object to modify.

        Raises:
            ValueError: If GPU allocation requirements cannot be satisfied based on
                       the current mode and available resources.
        """
        cluster = config.cluster
        if config.mode != "train":
            cluster.rollout_gpu_num = (
                config.explorer.rollout_model.tensor_parallel_size
                * config.explorer.rollout_model.engine_num
            )
            cluster.auxiliary_model_gpu_num = sum(
                model.tensor_parallel_size * model.engine_num
                for model in config.explorer.auxiliary_models
            )
        cluster.explorer_gpu_num = cluster.rollout_gpu_num + cluster.auxiliary_model_gpu_num
        cluster.total_gpu_num = cluster.node_num * cluster.gpu_per_node
        if config.mode in ["explore", "bench", "serve"]:
            if cluster.explorer_gpu_num > cluster.total_gpu_num:
                raise ValueError(
                    f"Total GPU number ({cluster.total_gpu_num}) is less than "
                    f"the number of GPUs required for rollout ({cluster.explorer_gpu_num})."
                )
        else:
            if cluster.explorer_gpu_num >= cluster.total_gpu_num:
                raise ValueError(
                    "Not enough GPUs for trainer. "
                    f"Explorer requires {cluster.explorer_gpu_num} GPUs, "
                    f"but total available GPUs are {cluster.total_gpu_num}."
                )

            cluster.trainer_gpu_num = cluster.total_gpu_num - cluster.explorer_gpu_num
            if cluster.trainer_gpu_num <= cluster.gpu_per_node:
                cluster.trainer_node_num = 1
                cluster.trainer_gpu_num_per_node = cluster.trainer_gpu_num
            else:
                if cluster.trainer_gpu_num % cluster.gpu_per_node != 0:
                    raise ValueError(
                        "Trainer must use an integer number of nodes, "
                        f"but got trainer_gpu_num ({cluster.trainer_gpu_num}) "
                        f"with gpu_per_node ({cluster.gpu_per_node}). "
                        "Please change `engine_num` or `tensor_parallel_size` in explorer config."
                    )
                cluster.trainer_node_num = cluster.trainer_gpu_num // cluster.gpu_per_node
                cluster.trainer_gpu_num_per_node = cluster.gpu_per_node


class AlgorithmConfigValidator(ConfigValidator):
    """Validator for algorithm-specific configuration.

    Handles algorithm type validation, sets default configuration parameters,
    validates function registry entries, and manages deprecated optimizer settings.
    """

    def validate(self, config: Config) -> None:
        """Validate and configure algorithm-specific settings.

        - Validates the algorithm type and runs algorithm-specific validation
        - Sets default configuration values for various algorithm components
        - Validates and configures function registry entries (loss functions, etc.)
        - Handles deprecated optimizer configuration parameters

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If invalid algorithm types or function names are specified.
        """
        from trinity.algorithm import (
            ADVANTAGE_FN,
            ALGORITHM_TYPE,
            ENTROPY_LOSS_FN,
            KL_FN,
            POLICY_LOSS_FN,
            SAMPLE_STRATEGY,
        )

        algorithm = ALGORITHM_TYPE.get(config.algorithm.algorithm_type)
        algorithm.check_config(config)
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
            set_if_none(config.algorithm, key, value)

        def check_and_set(name, registry, args_attr):
            fn_cls = registry.get(getattr(config.algorithm, name))
            if fn_cls is None:
                raise ValueError(f"Invalid {name}: {getattr(config.algorithm, name)}")
            set_if_none(config.algorithm, args_attr, fn_cls.default_args())
            return fn_cls

        check_and_set("sample_strategy", SAMPLE_STRATEGY, "sample_strategy_args")
        check_and_set("policy_loss_fn", POLICY_LOSS_FN, "policy_loss_fn_args")
        check_and_set("advantage_fn", ADVANTAGE_FN, "advantage_fn_args")
        check_and_set("kl_loss_fn", KL_FN, "kl_loss_fn_args")
        check_and_set("kl_penalty_fn", KL_FN, "kl_penalty_fn_args")
        check_and_set("entropy_loss_fn", ENTROPY_LOSS_FN, "entropy_loss_fn_args")
        if "loss_agg_mode" in config.algorithm.policy_loss_fn_args:  # type: ignore [operator]
            # override loss_agg_mode in policy_loss_fn_args
            config.algorithm.policy_loss_fn_args["loss_agg_mode"] = config.algorithm.loss_agg_mode  # type: ignore [index]

        optim_config = config.algorithm.optimizer
        if optim_config.warmup_style is not None:
            optim_config.lr_scheduler_type = optim_config.warmup_style
            self.logger.warning(
                "`warmup_style` is deprecated. Please use `lr_scheduler_type` instead. "
                f"And `lr_scheduler_type` is set to {optim_config.lr_scheduler_type}."
            )


class ModelConfigValidator(ConfigValidator):
    """Validator for model configuration settings.

    Handles model path validation, chat template loading, Tinker-specific validation,
    and model length parameter validation including prompt/response token limits.
    """

    def validate(self, config: Config) -> None:
        """Validate and configure model-specific settings.

        - Sets critic model path to actor model path if not specified
        - Loads chat templates from file if path is provided
        - Validates Tinker-specific configuration if enabled
        - Validates and sets model length parameters (max_model_len, max_prompt_tokens, etc.)

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If chat template file cannot be read, model length constraints
                       are violated, or Tinker configuration is invalid.
        """
        model = config.model
        if not model.critic_model_path:
            model.critic_model_path = model.model_path

        if model.tinker.enable:
            self._check_tinker(config)

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
        self._check_model_len(config)

    def _check_tinker(self, config: Config) -> None:
        """Validate Tinker-specific configuration settings.

        - Validates that critic models are not used with Tinker
        - Checks that the model is supported by the Tinker service
        - Issues warnings about entropy coefficient recommendations
        - Forces engine types to 'tinker' for all components
        - Disables NCCL synchronization for Tinker

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If critic models are used with Tinker or if the model
                       is not supported by the Tinker service.
        """
        model = config.model
        from trinity.algorithm import ALGORITHM_TYPE

        algorithm = ALGORITHM_TYPE.get(config.algorithm.algorithm_type)
        if algorithm.use_critic:
            raise ValueError("Critic model is not supported when using tinker!")

        import tinker

        service_client = tinker.ServiceClient()
        supported_models = {
            item.model_name for item in service_client.get_server_capabilities().supported_models
        }
        if model.model_path not in supported_models:
            self.logger.error(f"Supported models: {supported_models}")
            raise ValueError(f"{model.model_path} is not supported by tinker!")

        if (
            config.algorithm.entropy_loss_fn != "none"
            and config.algorithm.entropy_loss_fn_args.get("entropy_coef", 0.0) != 0.0
        ):
            self.logger.warning(
                "The entropy in Tinker trainer is an estimated value; "
                "it is recommended to set `entropy_coef` to 0."
            )

        if config.explorer.rollout_model.engine_type != "tinker":
            config.explorer.rollout_model.engine_type = "tinker"
            self.logger.warning("Rollout model engine type is set to `tinker`.")

        for aux_model_config in config.explorer.auxiliary_models:
            if aux_model_config.engine_type != "tinker":
                aux_model_config.engine_type = "tinker"
                self.logger.warning("Auxiliary model engine type is set to `tinker`.")

        if config.trainer.trainer_type != "tinker":
            config.trainer.trainer_type = "tinker"
            self.logger.warning("Trainer type is set to `tinker`.")

        if config.synchronizer.sync_method == SyncMethod.NCCL:
            config.synchronizer.sync_method = SyncMethod.CHECKPOINT
            self.logger.warning(
                "Tinker do not support NCCL, `synchronizer.sync_method` is set to `checkpoint`."
            )

    def _check_model_len(self, config: Config) -> None:
        """Validate and set model length configuration parameters.

        Ensures that max_model_len, max_prompt_tokens, and max_response_tokens
        are properly configured and consistent with each other. Sets defaults
        when values are missing and validates constraints.

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If model length constraints cannot be satisfied or
                       if required parameters are missing.
        """
        model = config.model
        # if all three are set, check if they are valid
        if (
            model.max_model_len is not None
            and model.max_prompt_tokens is not None
            and model.max_response_tokens is not None
        ):
            if model.max_prompt_tokens + model.max_response_tokens > model.max_model_len:
                raise ValueError(
                    "`max_prompt_tokens` + `max_response_tokens` "
                    f"({model.max_prompt_tokens} + {model.max_response_tokens}) "
                    f"exceeds `max_model_len` ({model.max_model_len}). "
                    "Please adjust them accordingly."
                )

        # check max_model_len first
        if model.max_model_len is None:
            if model.max_prompt_tokens is not None and model.max_response_tokens is not None:
                model.max_model_len = model.max_prompt_tokens + model.max_response_tokens
                self.logger.warning(
                    f"`max_model_len` is set to {model.max_model_len} from "
                    "`max_prompt_tokens` and `max_response_tokens`."
                )
            else:
                raise ValueError("Unable to determine `max_model_len`, please set it manually.")

        # both max_prompt_tokens and max_response_tokens are None
        if model.max_prompt_tokens is None and model.max_response_tokens is None:
            # default to max_model_len / 2
            model.max_prompt_tokens = model.max_model_len // 2
            model.max_response_tokens = model.max_model_len - model.max_prompt_tokens
            self.logger.warning(
                "`max_prompt_tokens` and `max_response_tokens` are not set, "
                f"set to {model.max_prompt_tokens} and {model.max_response_tokens} respectively."
            )

        # only max_prompt_tokens is None
        if model.max_prompt_tokens is None and model.max_response_tokens is not None:
            model.max_response_tokens = min(model.max_response_tokens, model.max_model_len - 1)
            model.max_prompt_tokens = model.max_model_len - model.max_response_tokens
            self.logger.warning(
                f"`max_prompt_tokens` is set to {model.max_prompt_tokens}, "
                f"`max_response_tokens` is set to {model.max_response_tokens}."
            )

        # only max_response_tokens is None
        if model.max_response_tokens is None and model.max_prompt_tokens is not None:
            model.max_prompt_tokens = min(model.max_prompt_tokens, model.max_model_len - 1)
            model.max_response_tokens = model.max_model_len - model.max_prompt_tokens
            self.logger.warning(
                f"`max_response_tokens` is set to {model.max_response_tokens}, "
                f"`max_prompt_tokens` is set to {model.max_prompt_tokens}."
            )

        if model.min_response_tokens >= model.max_response_tokens:  # type: ignore [operator]
            model.min_response_tokens = max(model.max_response_tokens - 1, 0)  # type: ignore [operator]
            self.logger.warning(f"`min_response_tokens` is set to {model.min_response_tokens}.")

        if model.enable_prompt_truncation is True:
            if model.max_prompt_tokens is None:
                raise ValueError(
                    "When `model.enable_prompt_truncation` is True, "
                    "`model.max_prompt_tokens` must be set properly. "
                    "This function does not work with OpenAI API mode."
                )
            self.logger.warning(
                "`enable_prompt_truncation` is set to True; the prompt will be"
                f" truncated to `max_prompt_tokens`={model.max_prompt_tokens} "
                "tokens if it is too long."
            )
        else:
            self.logger.warning(
                "`enable_prompt_truncation` is set to False; please make sure "
                "the prompt is not too long and `max_model_len` is large enough, "
                "otherwise prompt length + response length may exceed `max_model_len`!"
            )


class ExplorerConfigValidator(ConfigValidator):
    """Validator for explorer configuration settings.

    Handles rollout model configuration inheritance, auxiliary model validation,
    over-rollout ratio validation, and LoRA configuration processing.
    """

    def validate(self, config: Config) -> None:
        """Validate and configure explorer-specific settings.

        - Inherits model configuration from the global model config to rollout models
        - Validates auxiliary model configurations
        - Validates over-rollout ratio settings and compatibility with sync style
        - Processes LoRA configurations including dummy LoRA creation

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If auxiliary models lack model paths, over-rollout ratio
                       is invalid, or multiple LoRA adapters are configured.
        """
        if config.explorer is None:
            return

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

        # rollout model
        for args in model_args + ["model_path"]:
            set_if_none(config.explorer.rollout_model, args, getattr(config.model, args))
        set_if_none(
            config.explorer.rollout_model, "chat_template", config.model.custom_chat_template
        )

        # auxiliary models
        for aux_model in config.explorer.auxiliary_models:
            if not aux_model.model_path:
                raise ValueError("auxiliary model's model_path is required.")
            for args in model_args:
                set_if_none(aux_model, args, getattr(config.model, args))

        if config.explorer.over_rollout.ratio > 0.0:
            if not (0.0 <= config.explorer.over_rollout.ratio < 1.0):
                raise ValueError("over_rollout_ratio should be in [0.0, 1.0)")
            if config.synchronizer.sync_style == SyncStyle.FIXED:
                raise ValueError(
                    "over_rollout_ratio is not compatible with fixed sync_style, please set "
                    "`synchronizer.sync_style` to `dynamic_by_explorer` or `dynamic_by_trainer`."
                )

        self._validate_lora(config)

    def _validate_lora(self, config: Config) -> None:
        """Process and validate LoRA configuration settings.

        - Enables LoRA for rollout models when LoRA configs are provided
        - Validates that only one LoRA adapter is supported
        - Creates dummy LoRA adapters when no path is provided
        - Configures LoRA modules and kwargs for the rollout model

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If more than one LoRA adapter is configured.
        """
        # for lora configs
        if not config.model.tinker.enable and config.model.lora_configs is not None:
            config.explorer.rollout_model.enable_lora = True
            if len(config.model.lora_configs) > 1:
                raise ValueError("Only one lora adapter is supported for now.")
            lora_config = config.model.lora_configs[0]
            if lora_config.path is None:
                self.logger.info("Creating dummy lora, since no lora_path is provided.")
                lora_path = create_dummy_lora(
                    model_path=config.model.model_path,
                    checkpoint_job_dir=config.checkpoint_job_dir,
                    lora_rank=lora_config.lora_rank,
                    lora_alpha=lora_config.lora_alpha,
                    target_modules=lora_config.target_modules,
                    exclude_modules=lora_config.exclude_modules,
                )
                lora_config.path = lora_path
                lora_config.is_dummy = True
            config.explorer.rollout_model.lora_modules = [
                {
                    "lora_int_id": i + 1,
                    "lora_name": cfg.name,
                    "lora_path": cfg.path,
                    "base_model_name": cfg.base_model_name,
                }
                for i, cfg in enumerate(config.model.lora_configs)
            ]
            config.explorer.rollout_model.lora_kwargs = {
                "max_loras": len(config.model.lora_configs),
                "max_lora_rank": max(
                    (
                        model_config.lora_rank
                        for model_config in config.model.lora_configs
                        if model_config.lora_rank > 0
                    ),
                    default=0,
                ),
                "default_lora_path": os.path.join(
                    config.checkpoint_job_dir, "global_step_0", "actor", "lora_adapter"
                ),  # will be poped later
            }


class SynchronizerConfigValidator(ConfigValidator):
    """Validator for synchronizer configuration settings.

    Handles synchronizer namespace configuration and validates NCCL synchronization
    compatibility with different modes and features.
    """

    def validate(self, config: Config) -> None:
        """Validate and configure synchronizer settings.

        - Sets the Ray namespace for the synchronizer
        - Sets the explorer world size based on rollout GPU count
        - Disables NCCL synchronization for incompatible modes and features

        Args:
            config: The global configuration object to validate.
        """
        config.synchronizer.ray_namespace = config.ray_namespace
        config.synchronizer.explorer_world_size = config.cluster.rollout_gpu_num
        if config.synchronizer.sync_method == SyncMethod.NCCL:
            if config.mode in ["train", "explore", "bench", "serve"]:
                config.synchronizer.sync_method = SyncMethod.CHECKPOINT
                self.logger.warning(
                    f"`{config.mode}` mode does not support NCCL synchronization, "
                    "set `synchronizer.sync_method` to `checkpoint`."
                )
            if config.model.lora_configs is not None:
                config.synchronizer.sync_method = SyncMethod.CHECKPOINT
                self.logger.warning(
                    "LoRA is not supported with NCCL synchronization, "
                    "set `synchronizer.sync_method` to `checkpoint`."
                )


class IntervalConfigValidator(ConfigValidator):
    """Validator for interval configuration settings.

    Validates synchronization and evaluation intervals, ensuring that evaluation
    intervals are multiples of synchronization intervals when applicable.
    """

    def validate(self, config: Config) -> None:
        """Validate interval configuration settings.

        - Ensures synchronization interval is positive
        - Adjusts evaluation interval to be a multiple of sync interval when needed

        Args:
            config: The global configuration object to validate.

        Raises:
            AssertionError: If synchronization interval is not positive.
        """
        assert config.synchronizer.sync_interval > 0, "`sync_interval` must be positive."

        if config.mode != "bench" and config.algorithm.algorithm_type != "dpo":  # TODO
            # check eval_interval
            if config.explorer.eval_interval % config.synchronizer.sync_interval != 0:
                config.explorer.eval_interval = (
                    max(config.explorer.eval_interval // config.synchronizer.sync_interval, 1)
                ) * config.synchronizer.sync_interval
                self.logger.warning(
                    "`eval_interval` is not a multiple of `sync_interval`; "
                    f"adjusted to the nearest integer={config.explorer.eval_interval}."
                )


class MonitorConfigValidator(ConfigValidator):
    """Validator for monitor configuration settings.

    Validates monitor type, sets default arguments, and configures monitor cache directory.
    """

    def validate(self, config: Config) -> None:
        """Validate and configure monitor settings.

        - Validates that the monitor type is supported
        - Sets default monitor arguments if not provided
        - Creates the monitor cache directory

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If an invalid monitor type is specified.
        """
        from trinity.utils.monitor import MONITOR

        monitor_cls = MONITOR.get(config.monitor.monitor_type)
        if monitor_cls is None:
            raise ValueError(f"Invalid monitor type: {config.monitor.monitor_type}")
        set_if_none(config.monitor, "monitor_args", monitor_cls.default_args())
        # create a job dir in <checkpoint_root_dir>/<project>/<name>/monitor
        config.monitor.cache_dir = os.path.join(config.checkpoint_job_dir, "monitor")
        try:
            os.makedirs(config.monitor.cache_dir, exist_ok=True)
        except Exception:
            self.logger.warning(
                f"Failed to create monitor dir {config.monitor.cache_dir}, please check "
                f"your checkpoint directory: {config.checkpoint_job_dir}"
            )


class BufferConfigValidator(ConfigValidator):
    """Validator for buffer configuration settings.

    Handles train batch size validation, buffer directory setup, tokenizer configuration,
    and comprehensive validation of explorer/trainer input configurations including
    tasksets, experience buffers, and data pipelines.
    """

    def validate(self, config: Config) -> None:
        """Validate and configure buffer settings.

        - Sets train batch size based on mode and algorithm configuration
        - Creates buffer cache directory
        - Configures pad token ID using tokenizer
        - Validates explorer input configurations (tasksets, selectors)
        - Validates trainer input configurations (experience buffers, auxiliary buffers)
        - Validates data processor pipeline configurations

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If required buffer configurations are missing or invalid.
            RuntimeError: If buffer directory creation fails.
        """
        # check train_batch_size
        if not config.buffer.train_batch_size:
            if config.mode == "train" or config.algorithm.algorithm_type in ["sft", "dpo"]:
                raise ValueError(
                    "`buffer.train_batch_size` is required when `mode` is 'train' "
                    "or `algorithm.algorithm_type` is 'sft' or 'dpo'"
                )
            self.logger.info(
                "`buffer.train_batch_size` is set to `buffer.batch_size` * `algorithm.repeat_times`"
            )
            config.buffer.train_batch_size = (
                config.buffer.batch_size * config.algorithm.repeat_times
            )
        if (
            config.mode in {"train", "both"}
            and config.buffer.train_batch_size % config.cluster.trainer_gpu_num != 0
        ):
            raise ValueError(
                f"batch_size ({config.buffer.train_batch_size}) must be "
                f"divisible by ({config.cluster.trainer_gpu_num})."
            )

        # create buffer.cache_dir at <checkpoint_root_dir>/<project>/<name>/buffer
        config.buffer.cache_dir = os.path.abspath(os.path.join(config.checkpoint_job_dir, "buffer"))
        try:
            os.makedirs(config.buffer.cache_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create buffer dir {config.buffer.cache_dir}, please check "
                f"your checkpoint directory: {config.checkpoint_job_dir}"
            ) from e

        # set pad_token_id / tokenizer_path
        if config.buffer.pad_token_id is None:
            from transformers import AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    self.logger.warning(
                        f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}",
                        stacklevel=1,
                    )
                config.buffer.pad_token_id = tokenizer.pad_token_id

            except Exception:
                self.logger.warning(
                    f"Failed to get pad token id from model {config.model.model_path}"
                )
                config.buffer.pad_token_id = 0

        self._check_explorer_input(config)
        self._check_trainer_input(config)
        self._check_data_processor(config)

    def _check_explorer_input(self, config: Config):
        """Validate explorer input configuration including tasksets and selectors.

        - Handles taskset vs tasksets configuration
        - Validates that at least one taskset is provided in non-bench modes
        - Configures taskset defaults and validates selectors

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If taskset configuration is invalid or selectors are unsupported.
        """
        from trinity.buffer.selector import SELECTORS

        if config.mode in {"train", "serve"}:
            # no need to check explorer_input in serve mode
            return

        explorer_input = config.buffer.explorer_input

        if explorer_input.taskset:
            if len(explorer_input.tasksets) > 0:
                raise ValueError("Do not support setting `taskset` and `tasksets` simultaneously!")
            explorer_input.tasksets = [explorer_input.taskset]
            explorer_input.taskset = None
        elif config.mode != "bench" and len(explorer_input.tasksets) == 0:
            raise ValueError("At least one taskset should be provided in explorer_input!")

        def _fill_taskset_config(taskset: TasksetConfig, index: int, is_eval: bool = False):
            if not taskset.path:
                prefix = "Eval taskset" if is_eval else "Train taskset"
                raise ValueError(f"{prefix} [{taskset}]'s path is not configured.")

            if not taskset.name:
                prefix = "eval_" if is_eval else ""
                taskset.name = f"{prefix}taskset_{index}"
            taskset.is_eval = is_eval

            taskset.batch_size = config.buffer.batch_size
            if not is_eval:
                taskset.total_epochs = config.buffer.total_epochs
                taskset.total_steps = config.buffer.total_steps
                if taskset.repeat_times != config.algorithm.repeat_times:
                    taskset.repeat_times = config.algorithm.repeat_times
                    self.logger.info(
                        "`buffer.explorer_input.taskset.repeat_times` is set to "
                        f"`algorithm.repeat_times` (={config.algorithm.repeat_times})."
                    )

            set_if_none(taskset, "default_workflow_type", explorer_input.default_workflow_type)
            set_if_none(taskset, "default_reward_fn_type", explorer_input.default_reward_fn_type)
            set_if_none(taskset, "ray_namespace", config.ray_namespace)
            for attr in ["temperature", "top_p", "top_k", "logprobs"]:
                set_if_none(taskset.rollout_args, attr, getattr(config.model, attr))
            set_if_none(taskset.rollout_args, "max_tokens", config.model.max_response_tokens)
            set_if_none(taskset.format, "chat_template", config.model.custom_chat_template)

        for i, taskset in enumerate(explorer_input.tasksets):
            _fill_taskset_config(taskset, i)

            # check if selector is supported
            selector = SELECTORS.get(taskset.task_selector.selector_type)
            if selector is None:
                raise ValueError(
                    f"Selector {taskset.task_selector.selector_type} is not supported."
                )

        for idx, taskset in enumerate(explorer_input.eval_tasksets):
            # eval_workflow has higher priority than workflow in eval tasksets, so we set it first
            set_if_none(taskset, "default_workflow_type", explorer_input.default_eval_workflow_type)
            _fill_taskset_config(taskset, idx, is_eval=True)

    def _check_trainer_input(self, config: Config):
        """Validate trainer input configuration including experience buffers.

        - Configures experience buffer defaults and storage types
        - Validates auxiliary buffer configurations
        - Sets buffer schema types based on algorithm

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If required trainer input configurations are missing.
            AssertionError: If experience buffer is missing in train mode.
        """
        if config.mode == "bench":
            # no need to check trainer_input in bench mode
            return

        trainer_input = config.buffer.trainer_input
        experience_buffer = trainer_input.experience_buffer

        if experience_buffer is None:
            experience_buffer = trainer_input.experience_buffer = ExperienceBufferConfig(
                name="experience_buffer",
                storage_type=StorageType.QUEUE.value,
            )
            self.logger.info(
                f"Auto set `buffer.trainer_input.experience_buffer` to {experience_buffer}"
            )
        elif experience_buffer.storage_type == StorageType.FILE.value and config.mode == "both":
            self.logger.warning(
                "`FILE` storage is not supported to use as experience_buffer "
                "in `both` mode, use `QUEUE` instead."
            )
            experience_buffer.storage_type = StorageType.QUEUE.value

        if not experience_buffer.name:
            experience_buffer.name = "experience_buffer"

        if not experience_buffer.path:
            experience_buffer.path = self._default_storage_path(
                config, experience_buffer.storage_type, experience_buffer.name
            )
            self.logger.warning(
                f"Auto set `buffer.trainer_input.experience_buffer.path` to {experience_buffer.path}"
            )

        from trinity.algorithm import ALGORITHM_TYPE

        experience_buffer.schema_type = ALGORITHM_TYPE.get(config.algorithm.algorithm_type).schema
        experience_buffer.batch_size = config.buffer.train_batch_size
        experience_buffer.tokenizer_path = config.model.model_path
        set_if_none(experience_buffer, "ray_namespace", config.ray_namespace)
        set_if_none(experience_buffer.format, "chat_template", config.model.custom_chat_template)
        for aux_name, aux_buffer in trainer_input.auxiliary_buffers.items():
            aux_buffer.batch_size = config.buffer.train_batch_size
            aux_buffer.tokenizer_path = config.model.model_path
            set_if_none(aux_buffer, "ray_namespace", config.ray_namespace)
            if aux_buffer.path is None or aux_buffer.path == "":
                raise ValueError(
                    f"`buffer.trainer_input.auxiliary_buffers[{aux_name}].path` is required, "
                    f"please set it to the path of the auxiliary buffer."
                )

        if config.mode == "train":
            assert (
                experience_buffer is not None
            ), "`buffer.trainer_input.experience_buffer` is required when `mode` is `train`."
            experience_buffer.total_epochs = config.buffer.total_epochs
            experience_buffer.total_steps = config.buffer.total_steps

    def _default_storage_path(self, config: Config, storage_type: str, name: str) -> str:
        """Generate default storage path based on storage type.

        Args:
            config: The global configuration object.
            storage_type: The type of storage (SQL, FILE, etc.).
            name: The name of the storage component.

        Returns:
            The default storage path for the given storage type and name.
        """
        if storage_type == StorageType.SQL.value:
            return "sqlite:///" + os.path.join(config.buffer.cache_dir, f"{name}.db")  # type: ignore[arg-type]
        else:
            return os.path.join(config.buffer.cache_dir, f"{name}.jsonl")  # type: ignore[arg-type]

    def _check_data_processor(self, config: Config):
        """Validate data processor pipeline configurations.

        - Configures experience pipeline input save paths
        - Integrates Data-Juicer service configuration into operators
        - Validates task pipeline output configuration and path conflicts

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If task pipeline output is missing or path already exists.
        """
        # check input/output buffers in pipelines
        experience_pipeline = config.data_processor.experience_pipeline
        if experience_pipeline is not None and config.mode in {"explore", "both", "serve"}:
            if experience_pipeline.save_input and experience_pipeline.input_save_path is None:
                experience_pipeline.input_save_path = self._default_storage_path(
                    config, StorageType.SQL.value, "explorer_output"
                )
                self.logger.info(
                    "Auto set `data_processor.experience_pipeline.input_save_path` "
                    f"to {experience_pipeline.input_save_path}"
                )

            if config.service.data_juicer is not None:
                for operator in experience_pipeline.operators:
                    if operator.name == "data_juicer":
                        operator.args["service_config"] = config.service.data_juicer

        task_pipeline = config.data_processor.task_pipeline
        if task_pipeline is not None and config.mode in {"explore", "train", "both"}:
            if task_pipeline.output is None:
                if config.mode != "train":
                    if len(config.buffer.explorer_input.tasksets) > 0:
                        task_pipeline.output = config.buffer.explorer_input.tasksets[0]
                    else:
                        raise ValueError(
                            "At least one taskset should be provided in explorer_input!"
                        )
                elif config.mode == "train" and config.algorithm.algorithm_type in {"dpo", "sft"}:
                    task_pipeline.output = config.buffer.trainer_input.experience_buffer
                else:
                    raise ValueError(
                        "`data_processor.task_pipeline.output` is missing. "
                        "Please set it to the desired output storage config."
                    )
            if task_pipeline.output.path and os.path.exists(task_pipeline.output.path):
                raise ValueError(
                    f"Task pipeline output path {task_pipeline.output.path} already exists.\n"
                    "Please choose a different output path to avoid overwriting."
                )


class TrainerConfigValidator(ConfigValidator):
    """Validator for trainer configuration settings.

    Handles trainer type validation, configuration merging, and parameter validation
    for different trainer implementations (veRL, Tinker, etc.).
    """

    def validate(self, config: Config) -> None:
        """Validate and configure trainer settings.

        - Validates trainer type and handles configuration for different trainer types
        - Merges trainer configuration with schema defaults
        - Validates save checkpoint strategy options
        - Synchronizes trainer configuration with global config

        Args:
            config: The global configuration object to validate.

        Raises:
            ValueError: If trainer type is invalid, deprecated config path is used,
                       or save checkpoint strategy is invalid.
        """
        if (
            config.mode not in ["train", "both", "bench"]
            and config.trainer.trainer_strategy != "megatron"
        ):
            return

        if config.trainer.trainer_type == "verl":
            if config.trainer.trainer_config:
                from trinity.common.verl_config import veRLConfig

                trainer_config_schema = OmegaConf.structured(veRLConfig)
                trainer_config = OmegaConf.merge(
                    trainer_config_schema, config.trainer.trainer_config
                )
                config.trainer.trainer_config = OmegaConf.to_object(trainer_config)
            elif config.trainer.trainer_config_path:
                raise ValueError(
                    "`trainer_config_path` is deprecated; please use `trainer_config` instead."
                )
            else:
                from trinity.common.verl_config import veRLConfig

                self.logger.info("`trainer_config` is not provided, using default trainer config.")
                config.trainer.trainer_config = veRLConfig()
            if config.trainer.max_token_len_per_gpu is None:
                config.trainer.max_token_len_per_gpu = math.ceil(
                    2 * config.model.max_model_len / config.trainer.ulysses_sequence_parallel_size  # type: ignore [operator]
                )
            if config.trainer.save_hf_checkpoint not in {"last", "always", "never"}:
                raise ValueError(
                    f"Invalid trainer.save_hf_checkpoint: {config.trainer.save_hf_checkpoint}, "
                    "must be one of 'last', 'always', or 'never'."
                )
            config.trainer.trainer_config.synchronize_config(config)
        elif config.trainer.trainer_type == "tinker":
            config.trainer.trainer_config = None
        else:
            raise ValueError(f"Invalid trainer type: {config.trainer.trainer_type}")


validators = [
    DeprecatedConfigValidator(),
    GlobalConfigValidator(),
    MonitorConfigValidator(),
    AlgorithmConfigValidator(),
    ModelConfigValidator(),
    RayClusterConfigValidator(),
    SynchronizerConfigValidator(),
    IntervalConfigValidator(),
    ExplorerConfigValidator(),
    BufferConfigValidator(),
    TrainerConfigValidator(),
]
