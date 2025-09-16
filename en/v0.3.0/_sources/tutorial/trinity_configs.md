(Configuration Guide)=
# Configuration Guide

This section provides a detailed description of the configuration files used in **Trinity-RFT**.

## Overview

The configuration for **Trinity-RFT** is defined in a `YAML` file and organized into multiple sections based on different modules. Here's an example of a basic configuration file:

```yaml
project: Trinity-RFT
name: example
mode: both
checkpoint_root_dir: /PATH/TO/CHECKPOINT
continue_from_checkpoint: true

algorithm:
  # Algorithm-related parameters
  ...
model:
  # Model-specific configurations
  ...
cluster:
  # Cluster node and GPU settings
  ...
buffer:
  # Data buffer configurations
  ...
explorer:
  # Explorer-related settings (rollout models, workflow runners)
  ...
trainer:
  # Trainer-specific parameters
  ...
synchronizer:
  # Model weight synchronization settings
  ...
monitor:
  # Monitoring configurations (e.g., WandB, TensorBoard or MLFlow)
  ...
service:
  # Services to use
  ...
data_processor:
  # Preprocessing data settings
  ...
log:
  # Ray actor logging
  ...
```

Each of these sections will be explained in detail below. For additional details about specific parameters not covered here, please refer to the [source code](https://github.com/modelscope/Trinity-RFT/blob/main/trinity/common/config.py).

```{tip}
Trinity-RFT uses [OmegaConf](https://omegaconf.readthedocs.io/en/latest/) to load YAML configuration files.
It supports some advanced features like [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) and  [environment variable substitution](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#oc-env).
Users can use these features to simplify configuration.
```

---

## Global Configuration

These are general settings that apply to the entire experiment.

```yaml
project: Trinity-RFT
name: example
mode: both
checkpoint_root_dir: ${oc.env:CHECKPOINT_ROOT_DIR}   # CHECKPOINT_ROOT_DIR is an environment variable set in advance
```

- `project`: The name of the project.
- `name`: The name of the current experiment.
- `mode`: Running mode of Trinity-RFT. Options include:
  - `both`: Launches both the trainer and explorer (default).
  - `train`: Only launches the trainer.
  - `explore`: Only launches the explorer.
  - `bench`: Used for benchmarking.
- `checkpoint_root_dir`: Root directory where all checkpoints and logs will be saved. Checkpoints for this experiment will be stored in `<checkpoint_root_dir>/<project>/<name>/`.
- `continue_from_checkpoint`: If set to `true`, the experiment will continue from the latest checkpoint in the checkpoint path (if any); otherwise, it will rename the current experiment to `<name>_<timestamp>` and start a new experiment.
- `ray_namespace`: Namespace for the modules launched in the current experiment. If not specified, it will be set to `<project>/<name>`.

---

## Algorithm Configuration

Specifies the algorithm type and its related hyperparameters.

```yaml
algorithm:
  algorithm_type: grpo
  repeat_times: 8

  # The following parameters are optional
  # If not specified, they will automatically be set based on the `algorithm_type`
  sample_strategy: "default"
  advantage_fn: "ppo"
  kl_penalty_fn: "none"
  kl_loss_fn: "k2"
  entropy_loss_fn: "default"
```

- `algorithm_type`: Type of reinforcement learning algorithm. Supported types: `ppo`, `grpo`, `opmd`, `dpo`, `sft`, `mix`.
- `repeat_times`: Number of times each task is repeated. Default is `1`. In `dpo`, this is automatically set to `2`. Some algorithms such as GRPO and OPMD require `repeat_times` > 1.
- `sample_strategy`: The sampling strategy used for loading experiences from experience buffer.
- `advantage_fn`: The advantage function used for computing advantages.
- `kl_penalty_fn`: The KL penalty function used for computing KL penalty applied in reward.
- `kl_loss_fn`: The KL loss function used for computing KL loss.
- `entropy_loss_fn`: The entropy loss function used for computing entropy loss.

---

## Monitor Configuration

Used to log training metrics during execution.

```yaml
monitor:
  monitor_type: wandb
  monitor_args:
    base_url: http://localhost:8080
    api_key: your_api_key
  enable_ray_timeline: False
```

- `monitor_type`: Type of monitoring system. Options:
  - `wandb`: Logs to [Weights & Biases](https://docs.wandb.ai/quickstart/). Requires logging in and setting `WANDB_API_KEY`. Project and run names match the `project` and `name` fields in global configs.
  - `tensorboard`: Logs to [TensorBoard](https://www.tensorflow.org/tensorboard). Files are saved under `<checkpoint_root_dir>/<project>/<name>/monitor/tensorboard`.
  - `mlflow`: Logs to [MLFlow](https://mlflow.org/). If [MLFlow authentication](https://mlflow.org/docs/latest/ml/auth/) is setup, set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` as environment variables before running.
- `monitor_args`: Dictionary of arguments for monitor initialization.
  - For `wandb`:
    - `base_url`: Overrides `WANDB_BASE_URL` if set.
    - `api_key`: Overrides `WANDB_API_KEY` if set.
  - For `mlflow`:
    - `uri`: The URI of your MLFlow instance. Strongly recommended to set; defaults to `http://localhost:5000`.
    - `username`: Overrides `MLFLOW_TRACKING_USERNAME` if set.
    - `password`: Overrides `MLFLOW_TRACKING_PASSWORD` if set.
- `enable_ray_timeline`: If `True`, exports a `timeline.json` file to `<checkpoint_root_dir>/<project>/<name>/monitor`. Viewable in Chrome at [chrome://tracing](chrome://tracing).

---

## Model Configuration

Defines the model paths and token limits.

```yaml
model:
  model_path: ${oc.env:MODEL_PATH}  # MODEL_PATH is an environment variable set in advance
  critic_model_path: ${model.model_path}  # use the value of model.model_path
  max_response_tokens: 16384
  max_model_len: 20480
```

- `model_path`: Path to the model being trained.
- `critic_model_path`: Optional path to a separate critic model. If empty, defaults to `model_path`.
- `max_response_tokens`: Maximum number of tokens allowed in generated responses.
- `max_model_len`: Maximum number of tokens in a sequence.

---

## Cluster Configuration

Defines how many nodes and GPUs per node are used.

```yaml
cluster:
  node_num: 1
  gpu_per_node: 8
```

- `node_num`: Total number of compute nodes.
- `gpu_per_node`: Number of GPUs available per node.

---

## Buffer Configuration

Configures the data buffers used by the explorer and trainer.

```yaml
buffer:
  batch_size: 32
  train_batch_size: 256
  total_epochs: 100

  explorer_input:
    taskset:
      ...
    eval_tasksets:
      ...
  trainer_input:
    experience_buffer:
      ...
    sft_warmup_dataset:
      ...

  default_workflow_type: 'math_workflow'
  default_reward_fn_type: 'countdown_reward'
```

- `batch_size`: Number of tasks used per training step. *Please do not multiply this value by the `algorithm.repeat_times` manually*.
- `train_batch_size`: Number of experiences used per training step. Defaults to `batch_size` * `algorithm.repeat_times`.
- `total_epochs`: Total number of training epochs.
- `total_steps`: Optional. The total number of training steps. If specified, `total_epochs` will be ignored.

### Explorer Input

Defines the dataset(s) used by the explorer for training and evaluation.

```yaml
buffer:
  explorer_input:
    taskset:
      name: countdown_train
      storage_type: file
      path: /PATH/TO/DATA
      split: train
      format:
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 1.0
      default_workflow_type: 'math_workflow'
      default_reward_fn_type: 'countdown_reward'

    eval_tasksets:
    - name: countdown_eval
      storage_type: file
      path: /PATH/TO/DATA
      split: test
      repeat_times: 1
      format:
        prompt_type: `plaintext`
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 0.1
      default_workflow_type: 'math_workflow'
      default_reward_fn_type: 'countdown_reward'
    ...
```

- `buffer.explorer_input.taskset`: Task dataset used for training exploration policies.
- `buffer.explorer_input.eval_taskset`: List of task datasets used for evaluation.

The configuration for each task dataset is defined as follows:

- `name`: Name of the dataset. This name will be used as the Ray actor's name, so it must be unique.
- `storage_type`: How the dataset is stored. Options: `file`, `queue`, `sql`.
  - `file`: The dataset is stored in `jsonl`/`parquet` files. The data file organization is required to meet the huggingface standard. *We recommand using this storage type for most cases.*
  - `queue`: The dataset is stored in a queue. The queue is a simple FIFO queue that stores the task dataset. *Do not use this storage type for task dataset unless you know what you are doing.*
  - `sql`: The dataset is stored in a SQL database. *This type is unstable and will be optimized in the future versions.*
- `path`: The path to the task dataset.
  - For `file` storage type, the path points to the directory that contains the task dataset files.
  - For `queue` storage type, the path is optional. You can back up the data in the queue by specifying a sqlite database path here.
  - For `sql` storage type, the path points to the sqlite database file.
- `subset_name`: The subset name of the task dataset. Default is `None`.
- `split`: The split of the task dataset. Default is `train`.
- `repeat_times`: The number of rollouts generated for a task. If not set, it will be automatically set to `algorithm.repeat_times` for `taskset`, and `1` for `eval_tasksets`.
- `rollout_args`: The parameters for rollout.
  - `temperature`: The temperature for sampling.
- `default_workflow_type`: Type of workflow logic applied to this dataset. If not specified, the `buffer.default_workflow_type` is used.
- `default_reward_fn_type`: Reward function used during exploration. If not specified, the `buffer.default_reward_fn_type` is used.
- `workflow_args`: A dictionary of arguments used to supplement dataset-level parameters.

### Trainer Input

Defines the experience buffer and optional SFT warm-up dataset.

```yaml
buffer:
  ...
  trainer_input:
    experience_buffer:
      name: countdown_buffer
      storage_type: queue
      path: sqlite:///countdown_buffer.db
      max_read_timeout: 1800

    sft_warmup_dataset:
      name: warmup_data
      storage_type: file
      path: /PATH/TO/WARMUP_DATA
      format:
        prompt_key: 'question'
        response_key: 'answer'

    sft_warmup_steps: 0
```

- `experience_buffer`: It is the input of Trainer and also the output of Explorer. This field is required even in explore mode.
  - `name`: The name of the experience buffer. This name will be used as the Ray actor's name, so it must be unique.
  - `storage_type`: The storage type for the experience buffer.
    - `queue`: Experience data is stored in a queue. This storage type is recommended for most use cases.
    - `sql`: Experience data is stored in a SQL database.
    - `file`: Experience data is stored in a JSON file. This storage type should be used only for debugging purposes in `explore` mode.
  - `path`: The path to the experience buffer.
    - For `queue` storage type, this field is optional. You can specify a SQLite database or JSON file path here to back up the queue data.
    - For `file` storage type, the path points to the directory containing the dataset files.
    - For `sql` storage type, the path points to the SQLite database file.
  - `format`: Defines keys for prompts and responses in the dataset.
    - `prompt_type`: Specifies the type of prompts in the dataset. We support `plaintext`, `messages` for now.
      - `plaintext`: The prompt is in string format.
      - `messages`: The prompt is organized as a message list.
    - `prompt_key`: Specifies which column in the dataset contains the user prompt data. Only for `plaintext`.
    - `response_key`: Specifies which column in the dataset contains the response data. Only for `plaintext`.
    - `system_prompt_key`: Specifies which column in the dataset contains the system prompt data. Only for `plaintext`.
    - `system_prompt`: Specifies the system prompt in string format. It has lower priority than `system_prompt_key`. Only for `plaintext`.
    - `messages_key`: Specifies which column in the dataset contains the messages data. Only for `messages`.
    - `tools_key`: Specifies which column in the dataset contains the tools data. Support both `plaintext` and `messages`, but the tool data should be organized as a list of dict.
    - `chosen_key`: Specifies which column in the dataset contains the DPO chosen data. Support both `plaintext` and `messages`, and the data type should be consistent with the prompt type.
    - `rejected_key`: Similar to `chosen_key`, but it specifies which column in the dataset contains the DPO rejected data.
    - `enable_concatenated_multi_turn`: Enable concatenated multi-turn SFT data preprocess. Only for `messages` and only take effect with SFT algorithm.
    - `chat_template`: Specifies the chat template in string format. If not provided, use `model.custom_chat_template`.
  - `max_read_timeout`: The maximum waiting time (in seconds) to read new experience data. If exceeded, an incomplete batch will be returned directly. Only take effect when `storage_type` is `queue`. Default is 1800 seconds (30 minutes).
  - `use_priority_queue`: Only take effect when `storage_type` is `queue`. If set to `True`, the queue will be a priority queue, which allows for prioritizing certain experiences over others. Default is `False`.
  - `reuse_cooldown_time`: Only take effect when `storage_type` is `queue` and `use_priority_queue` is `True`. If set, it specifies the cooldown time (in seconds) for reusing experiences. If not specified, the default value is `None`, meaning experiences can not be reused.
- `sft_warmup_dataset`: Optional dataset used for pre-training (SFT warmup). Its configuration is similar to the `experience_buffer`, but only for SFT usage.
- `sft_warmup_steps`: Number of steps to use SFT warm-up before RL begins.

---

## Explorer Configuration

Controls the rollout models and workflow execution.

```yaml
explorer:
  name: explorer
  runner_per_model: 8
  max_timeout: 900
  max_retry_times: 2
  env_vars: {}
  rollout_model:
    engine_type: vllm
    engine_num: 1
    tensor_parallel_size: 1
    enable_history: False
  auxiliary_models:
  - model_path: /PATH/TO/MODEL
    tensor_parallel_size: 1
  eval_interval: 100
  eval_on_startup: True
```

- `name`: Name of the explorer. This name will be used as the Ray actor's name, so it must be unique.
- `runner_per_model`: Number of parallel workflow runners per each rollout model.
- `max_timeout`: Maximum time (in seconds) for a workflow to complete.
- `max_retry_times`: Maximum number of retries for a workflow.
- `env_vars`: Environment variables to be set for every workflow runners.
- `rollout_model.engine_type`: Type of inference engine. For now, only `vllm_async` and `vllm` is supported, they have the same meaning and both use the asynchronous engine. In subsequent versions, only `vllm` may be retained for simplicity.
- `rollout_model.engine_num`: Number of inference engines.
- `rollout_model.tensor_parallel_size`: Degree of tensor parallelism.
- `rollout_model.enable_history`: Whether to enable model call history recording. If set to `True`, the model wrapper automatically records the return experiences of model calls. Please periodically extract the history via `extract_experience_from_history` to avoid out-of-memory issues. Default is `False`.
- `auxiliary_models`: Additional models used for custom workflows.
- `eval_interval`: Interval (in steps) for evaluating the model.
- `eval_on_startup`: Whether to evaluate the model on startup. More precisely, at step 0 with the original model, so it will not be triggered when restarting.
- `runner_num`: (*Deprecated*) Number of parallel workflow runners.

---

## Synchronizer Configuration

Controls how model weights are synchronized between trainer and explorer.

```yaml
synchronizer:
  sync_method: 'nccl'
  sync_interval: 10
  sync_offset: 0
  sync_timeout: 1200
```

- `sync_method`: Method of synchronization. Options:
  - `nccl`: Uses NCCL for fast synchronization. Supported for `both` mode.
  - `checkpoint`: Loads latest model from disk. Supported for `train`, `explore`, or `bench` mode.
- `sync_interval`: Interval (in steps) of model weight synchronization between trainer and explorer.
- `sync_offset`: Offset (in steps) of model weight synchronization between trainer and explorer. The explorer can run `sync_offset` steps before the trainer starts training.
- `sync_timeout`: Timeout duration for synchronization.

---

## Trainer Configuration

Specifies the backend and behavior of the trainer.

```yaml
trainer:
  name: trainer
  trainer_type: 'verl'
  save_interval: 100
  trainer_config_path: ''
  trainer_config: null
```

- `name`: Name of the trainer. This name will be used as the Ray actor's name, so it must be unique.
- `trainer_type`: Trainer backend implementation. Currently only supports `verl`.
- `save_interval`: Frequency (in steps) at which to save model checkpoints.
- `trainer_config_path`: The path to the trainer configuration file.
- `trainer_config`: The trainer configuration provided inline. Only one of `trainer_config_path` and `trainer_config` should be specified.

---

## Service Configuration

Configures services used by Trinity-RFT. Only support Data Juicer service for now.

```yaml
service:
  data_juicer:
    server_url: 'http://127.0.0.1:5005'
    auto_start: true
    port: 5005
```

- `server_url`: The url of data juicer server.
- `auto_start`: Whether to automatically start the data juicer service.
- `port`: The port for Data Juicer service when `auto_start` is true.

---

## DataProcessor Configuration

Configures the task / experience pipeline, please refer to {ref}`Data Processing <Data Processing>` section for details.

```yaml
data_processor:
  task_pipeline:
  # task pipeline related
  task_pipeline:
    num_process: 32
    operators:
      - name: "llm_difficulty_score_filter"
        args:
          api_or_hf_model: "qwen2.5-7b-instruct"
          min_score: 0.0
          input_keys: ["question", "answer"]
          field_names: ["Question", "Answer"]
    inputs:  # the output will be set to the explorer input automatically
      - /PATH/TO/GSM8K/DATA/FILE
    target_fields: ["question", "answer"]
  experience_pipeline:
    operators:
      - name: data_juicer
        args:
          config_path: 'examples/grpo_gsm8k_experience_pipeline/dj_scoring_exp.yaml'
      - name: reward_shaping_mapper
        args:
          reward_shaping_configs:
            - stats_key: 'llm_quality_score'
              op_type: ADD
              weight: 1.0
```

--

## Log Configuration

Ray actor logging configuration.

```yaml
log:
  level: INFO
  group_by_node: False
```

- `level`: The logging level (supports `DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `group_by_node`: Whether to group logs by node IP. If set to `True`, an actor's logs will be save to `<checkpoint_root_dir>/<project>/<name>/log/<node_ip>/<actor_name>.log`, otherwise it will be saved to `<checkpoint_root_dir>/<project>/<name>/log/<actor_name>.log`.

---

## veRL Trainer Configuration (Advanced)

For advanced users working with the `verl` trainer backend. This includes fine-grained settings for actor/critic models, optimizer parameters, and training loops.

> For full parameter meanings, refer to the [veRL documentation](https://verl.readthedocs.io/en/latest/examples/config.html).


```yaml
actor_rollout_ref:
  hybrid_engine: True
  model:
    external_lib: null
    override_config: { }
    enable_gradient_checkpointing: True
    use_remove_padding: True
    use_fused_kernels: False
    fused_kernel_options:
      impl_backend: None
  actor:
    strategy: fsdp  # This is for backward-compatibility
    # ppo_micro_batch_size: 8 # will be deprecated, use ppo_micro_batch_size_per_gpu
    ppo_micro_batch_size_per_gpu: 4
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 16384 # n * ${data.max_model_len}
    grad_clip: 1.0
    ppo_epochs: 1
    shuffle: False
    ulysses_sequence_parallel_size: 1 # sp size
    entropy_from_logits_with_chunking: false
    entropy_checkpointing: false
    checkpoint:
      load_contents: ['model', 'optimizer', 'extra']
      save_contents: ['model', 'optimizer', 'extra']
    optim:
      lr: 1e-6
      lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
      # min_lr_ratio: null   # only useful for warmup with cosine
      warmup_style: constant  # select from constant/cosine
      total_training_steps: -1  # must be override by program
    fsdp_config:
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
      param_offload: False
      optimizer_offload: False
      fsdp_size: -1
      forward_prefetch: False
  ref:
    fsdp_config:
      param_offload: False
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
      fsdp_size: -1
      forward_prefetch: False
    # log_prob_micro_batch_size: 4 # will be deprecated, use log_prob_micro_batch_size_per_gpu
    log_prob_micro_batch_size_per_gpu: 4
    log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
    log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
    ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} # sp size
    entropy_from_logits_with_chunking: ${actor_rollout_ref.actor.entropy_from_logits_with_chunking}
    entropy_checkpointing: ${actor_rollout_ref.actor.entropy_checkpointing}

critic:
  strategy: fsdp
  optim:
    lr: 1e-5
    lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
    # min_lr_ratio: null   # only useful for warmup with cosine
    warmup_style: constant  # select from constant/cosine
    total_training_steps: -1  # must be override by program
  model:
    override_config: { }
    external_lib: ${actor_rollout_ref.model.external_lib}
    enable_gradient_checkpointing: True
    use_remove_padding: False
    fsdp_config:
      param_offload: False
      optimizer_offload: False
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
      fsdp_size: -1
      forward_prefetch: False
  ppo_micro_batch_size_per_gpu: 8
  forward_micro_batch_size_per_gpu: ${critic.ppo_micro_batch_size_per_gpu}
  use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
  ppo_max_token_len_per_gpu: 32768 # (${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}) * 2
  forward_max_token_len_per_gpu: ${critic.ppo_max_token_len_per_gpu}
  ulysses_sequence_parallel_size: 1 # sp size
  ppo_epochs: ${actor_rollout_ref.actor.ppo_epochs}
  shuffle: ${actor_rollout_ref.actor.shuffle}
  grad_clip: 1.0
  cliprange_value: 0.5

trainer:
  balance_batch: True
  # total_training_steps: null
  # auto: find the last ckpt to resume. If can't find, start from scratch
  resume_mode: auto # or auto or resume_path if
  resume_from_path: ""
  critic_warmup: 0
  default_hdfs_dir: null
  remove_previous_ckpt_in_save: False
  del_local_ckpt_after_load: False
  val_before_train: False
  max_actor_ckpt_to_keep: 5
  max_critic_ckpt_to_keep: 5
```


- `actor_rollout_ref.model.enable_gradient_checkpointing`: Whether to enable gradient checkpointing, which will reduce GPU memory usage.
- `actor_rollout_ref.model.use_remove_padding`: Whether to remove pad tokens, which will reduce training time.
- `actor_rollout_ref.model.use_fused_kernels`: Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
- `actor_rollout_ref.model.fused_kernel_options.impl_backend`: Implementation backend for fused kernels. If use_fused_kernels is true, this will be used. Options: "triton" or "torch".
- `actor_rollout_ref.actor.use_dynamic_bsz`: Whether to reorganize the batch data, specifically to splice the shorter data to reduce the batch size in the actual training process.
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: Batch size for one GPU in one forward pass.
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size`: Ulysses sequence parallel size.
- `actor_rollout_ref.actor.entropy_from_logits_with_chunking`: Calculate entropy with chunking to reduce memory peak.
- `actor_rollout_ref.actor.entropy_checkpointing`: Recompute entropy.
- `actor_rollout_ref.actor.checkpoint`: Contents to be loaded and saved. With 'hf_model' you can save whole model as hf format; now only use sharded model checkpoint to save space.
- `actor_rollout_ref.actor.optim.lr`: Learning rate for actor model.
- `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio`: Ratio of warmup steps for learning rate.
- `actor_rollout_ref.actor.optim.warmup_style`: Warmup style for learning rate.
- `actor_rollout_ref.actor.optim.total_training_steps`: Total training steps for actor model.
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: Batch size for one GPU in one reference model forward pass.

- `critic.model.enable_gradient_checkpointing`: Whether to enable gradient checkpointing, which will reduce GPU memory usage.
- `critic.model.use_remove_padding`: Whether to remove pad tokens, which will reduce training time.
- `critic.optim.lr`: Learning rate for critic model.
- `critic.optim.lr_warmup_steps_ratio`: Ratio of warmup steps for learning rate.
- `critic.optim.warmup_style`: Warmup style for learning rate.
- `critic.optim.total_training_steps`: Total training steps for critic model.
- `critic.ppo_micro_batch_size_per_gpu`: Batch size for one GPU in one critic model forward pass.
- `critic.ulysses_sequence_parallel_size`: Ulysses sequence parallel size.
- `critic.grad_clip`: Gradient clip for critic model training.
- `critic.cliprange_value`: Used for compute value loss.

- `trainer.balance_batch`: Whether to balance batch size between GPUs during training.
- `trainer.resume_mode`: Resume mode for training. Support `disable`, `auto` and `resume_path`.
- `trainer.resume_from_path`: Path to resume from.
- `trainer.critic_warmup`: The number of steps to train the critic model before actual policy learning.
- `trainer.default_hdfs_dir`: Default HDFS directory for saving checkpoints.
- `trainer.remove_previous_ckpt_in_save`: Whether to remove previous checkpoints in save.
- `trainer.del_local_ckpt_after_load`: Whether to delete local checkpoints after loading.
- `trainer.max_actor_ckpt_to_keep`: Maximum number of actor checkpoints to keep.
- `trainer.max_critic_ckpt_to_keep`: Maximum number of critic checkpoints to keep.
