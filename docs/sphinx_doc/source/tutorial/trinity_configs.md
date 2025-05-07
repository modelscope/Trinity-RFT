# Trinity-RFT Configuration

The following is the main config file for Trinity-RFT. Take `countdown.yaml` as an example.


## Monitor

```yaml
monitor:
  project: "Trinity-RFT-countdown"
  name: "qwen2.5-1.5B-countdown"
```

- `monitor.project`: The project name. It must be set manually.
- `monitor.name`: The name of the experiment. It must be set manually.


## Data

<!-- The `data` configuration specifies the data used for training. It includes the total number of epochs, the batch size, the path to the dataset, the default workflow type, the default reward function type, and the format configuration. -->

```yaml
data:
  dataset_path: '/PATH/TO/DATASET'
  train_split: 'train'
  eval_split: ''
  dataset_config:
    split: 'train'
  format_config:
    prompt_key: 'question'
    response_key: 'answer'

  db_url: ''
  max_retry_times: 3
  max_retry_interval: 1

  total_epochs: 20
  batch_size: 96
  default_workflow_type: 'math_workflow'
  default_reward_fn_type: 'countdown_reward'
```

- `data.dataset_path`: The path to the dataset.
- `data.train_split`: The split name of the dataset used for training. Default is `train`.
- `data.eval_split`: The split name of the dataset used for eval.
- `data.dataset_config`: The configuration for the dataset. <!-- TODO: may only used in Data-Juicer -->
- `data.format_config`: The configuration for the format of the dataset.
- `data.db_url`: The URL of the database.
- `data.max_retry_times`: The maximum number of retries when loading the dataset from database.
- `data.max_retry_interval`: The maximum interval between retries when loading the dataset from database.
- `data.total_epochs`: The total number of epochs to explore the dataset. Default is `1`. It should be set manually.
- `data.batch_size`: The number of `Task` in one training batch. The real batch size used in training is `data.batch_size` * `explorer.repeat_times`. It should be set manually.
- `data.default_workflow_type`: The default workflow type used for training.
- `data.default_reward_fn_type`: The default reward function type used for training.

<!-- TODO explain the dataset_config and format_config -->

## Model

The `model` configuration specifies the model used for training. It includes the path to the model checkpoint, the maximum number of tokens in the prompt, the maximum number of tokens in the response, the path to the checkpoint of the model, and whether to load the checkpoint of the model.

```yaml
model:
  model_path: '/PATH/TO/MODEL/CHECKPOINT/'
  critic_model_path: ''
  max_prompt_tokens: 256
  max_response_tokens: 1024
  checkpoint_path: 'checkpoints/qwen2.5-1.5B-countdown'
```

- `model.model_path`: The path to the model checkpoint. It must be set manually.
- `model.critic_model_path`: The path to the critic model checkpoint. If not set, the `model.critic_model_path` will be set to `model.model_path`.
- `model.max_prompt_tokens`: The maximum number of tokens in the prompt. Default is `2048`. It should be set manually.
- `model.max_response_tokens`: The maximum number of tokens in the response. Default is `2048`. It should be set manually.
- `model.checkpoint_path`: The path to the checkpoint of the model. It must be set manually.

## Cluster

The `cluster` configuration specifies the cluster configuration. It includes the number of nodes and the number of GPUs per node.

```yaml
cluster:
  node_num: 1
  gpu_per_node: 8
```

- `cluster.node_num`: The number of nodes used for training.
- `cluster.gpu_per_node`: The number of GPUs per node used for training.

## Buffer

```yaml
buffer:
  max_retry_times: 3
  max_retry_interval: 1
  train_dataset:
    name: countdown_buffer
    storage_type: queue
    algorithm_type: ppo
    path: 'sqlite:///countdown.db'
  sft_warmup_dataset: null
```

- `buffer.max_retry_times`: The maximum number of retries when loading the dataset from database.
- `buffer.max_retry_interval`: The maximum interval between retries when loading the dataset from database.
- `buffer.train_dataset`: The configuration of the training dataset.
- `buffer.sft_warmup_dataset`: The configuration of the SFT warmup dataset.

## Explorer

The `explorer` configuration specifies the explorer configuration. It includes the type of the engine, the number of engines, the number of workflow runners, the tensor parallel size, whether to enable prefix caching, whether to enforce eager mode, the data type, the `temperature`, the `top-p`, the `top-k`, the `seed`, the `logprobs`, the number of times to repeat each task, whether to use Ray, the backend, the maximum number of pending requests, and the maximum number of waitingsteps.

```yaml
explorer:
  engine_type: vllm_async
  engine_num: 2
  runner_num: 32
  tensor_parallel_size: 1
  enable_prefix_caching: false
  enforce_eager: true
  dtype: bfloat16
  temperature: 1.0
  seed: 42
  logprobs: 0
  repeat_times: 5
  use_ray: false
  backend: 'nccl'
  max_pending_requests: 32
  max_waiting_steps: 4
```

- `explorer.engine_type`: The type of the engine, Support `vllm_async` and `vllm_sync`. Default is `vllm_async`.
- `explorer.engine_num`: The number of engines. Default is `2`. It should be set manually.
- `explorer.runner_num`: The number of workflow runners. Default is `32`.
- `explorer.tensor_parallel_size`: The tensor parallel size used in vLLM. Default is `1`.
- `explorer.enable_prefix_caching`: Whether to enable prefix caching. Default is `False`.
- `explorer.enforce_eager`: Whether to enforce eager mode. Default is `True`.
- `explorer.dtype`: The data type used in vLLM. Default is `bfloat16`.
- `explorer.temperature`: The temperature used in vLLM. Default is `1.0`.
- `explorer.seed`: The seed used in vLLM. Default is `42`.
- `explorer.logprobs`: The logprobs used in vLLM. Default is `0`.
- `explorer.repeat_times`: The number of times to repeat each task, used for GRPO-like algorithms. Default is `5`.
- `explorer.use_ray`: Whether to use Ray. Default is `False`.
- `explorer.backend`: The backend used in vLLM. Default is `nccl`.
- `explorer.max_pending_requests`: The maximum number of pending requests. Default is `32`.
- `explorer.max_waiting_steps`: The maximum number of waiting steps. Default is `4`.

## Synchronizer

```yaml
synchronizer:
  sync_method: 'nccl'
  sync_interval: 10
  sync_timeout: 1200
```

- `synchronizer.sync_method`: The synchronization method between `trainer` and `explorer`.
Support `nccl` and `checkpoint`, `nccl` represents that model weights in `explorer` will be synchronized from `trainer` through `nccl`,
`checkpoint` represents that `explorer` will load the newest checkpoints saved by `trainer` then update its model weights. Default is `nccl`.
- `synchronizer.sync_interval`: The interval between two synchronizations. Default is `10`. It should be set manually.
- `synchronizer.sync_timeout`: The timeout of the synchronization. Default is `1200`.

## Trainer

```yaml
trainer:
  trainer_type: 'verl'
  algorithm_type: ppo
  trainer_config_path: 'examples/ppo_countdown/train_countdown.yaml'
  sft_warmup_steps: 0
  eval_interval: 1000
  save_interval: 100
```

- `trainer.trainer_type`: The backend of the trainer, Only `verl` is supported.
- `trainer.algorithm_type`: The type of the algorithm, Support `ppo`, `grpo`, `opmd` and `dpo`.
- `trainer.trainer_config_path`: The path to the trainer configuration file. It must be set manually.
- `trainer.sft_warmup_steps`: The number of steps to warm up the model. Default is `0`.
- `trainer.eval_interval`: The interval between two evaluations. Default is `1000`.
- `trainer.save_interval`: The interval between two checkpoints. Default is `100`.

### veRL Trainer Configuration

Here we mainly introduce the parameters that can be set in veRL. For the specific meaning of the parameters, please refer to the official document of [veRL](https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/docs/examples/config.rst).

```yaml
data:
  tokenizer: null
  train_files: train_example.parquet
  val_files: test_example.parquet
  prompt_key: prompt
  max_prompt_length: 256
  max_response_length: 1024
  train_batch_size: 256
  val_batch_size: null
  return_raw_input_ids: False  # This should be set to true when the tokenizer between policy and rm differs
  return_raw_chat: False
  shuffle: True
  filter_overlong_prompts: False # for large-scale dataset, filtering overlong prompts could be timeconsuming. You should disable this and set `truncation='left'
  truncation: error
  image_key: images

actor_rollout_ref:
  hybrid_engine: True
  model:
    path: /PATH/TO/MODEL/CHECKPOINT/
    external_lib: null
    override_config: { }
    enable_gradient_checkpointing: True
    use_remove_padding: True
  actor:
    strategy: fsdp  # This is for backward-compatibility
    ppo_mini_batch_size: 128
    # ppo_micro_batch_size: 8 # will be deprecated, use ppo_micro_batch_size_per_gpu
    ppo_micro_batch_size_per_gpu: 4
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 16384 # n * ${data.max_prompt_length} + ${data.max_response_length}
    grad_clip: 1.0
    clip_ratio: 0.2
    entropy_coeff: 0.001
    use_kl_loss: False # True for GRPO
    kl_loss_coef: 0.001 # for grpo
    kl_loss_type: low_var_kl # for grpo
    ppo_epochs: 1
    shuffle: False
    ulysses_sequence_parallel_size: 1 # sp size
    checkpoint:
      contents: ['model', 'hf_model', 'optimizer', 'extra']  # with 'hf_model' you can save whole model as hf format, now only use sharded model checkpoint to save space
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
    # --- below: opmd ---
    tau: 0.000  # strength of regularization w.r.t. old / ref policy
    opmd_baseline: mean  # mean / logavgexp, applicable to opmd
    use_uid: False  # True / False, applicable to pairwise_opmd
  ref:
    fsdp_config:
      param_offload: False
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
    # log_prob_micro_batch_size: 4 # will be deprecated, use log_prob_micro_batch_size_per_gpu
    log_prob_micro_batch_size_per_gpu: 8
    log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
    log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
    ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} # sp size
  rollout:
    name: vllm
    temperature: 1.0
    top_k: -1 # 0 for hf rollout, -1 for vllm rollout
    top_p: 1
    use_fire_sampling: False # https://arxiv.org/abs/2410.21236
    prompt_length: ${data.max_prompt_length}  # not use for opensource
    response_length: ${data.max_response_length}
    # for vllm rollout
    dtype: bfloat16 # should align with FSDP
    gpu_memory_utilization: 0.4
    ignore_eos: False
    enforce_eager: True
    free_cache_engine: True
    load_format: dummy_dtensor
    tensor_model_parallel_size: 2
    max_num_batched_tokens: 8192
    max_model_len: null
    max_num_seqs: 1024
    # log_prob_micro_batch_size: 8 # will be deprecated, use log_prob_micro_batch_size_per_gpu
    log_prob_micro_batch_size_per_gpu: 4
    log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
    log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
    disable_log_stats: True
    enable_chunked_prefill: True # could get higher throughput
    # for hf rollout
    do_sample: True
    # number of responses (i.e. num sample times)
    n: 1 # > 1 for grpo

critic:
  strategy: fsdp
  optim:
    lr: 1e-5
    lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
    # min_lr_ratio: null   # only useful for warmup with cosine
    warmup_style: constant  # select from constant/cosine
    total_training_steps: -1  # must be override by program
  model:
    path: /PATH/TO/MODEL/CHECKPOINT/
    tokenizer_path: ${actor_rollout_ref.model.path}
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
  ppo_mini_batch_size: ${actor_rollout_ref.actor.ppo_mini_batch_size}
  # ppo_micro_batch_size: 8 # will be deprecated, use ppo_micro_batch_size_per_gpu
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

reward_model:
  enable: False
  strategy: fsdp
  model:
    input_tokenizer: ${actor_rollout_ref.model.path}  # set this to null if the chat template is identical
    path: ~/models/FsfairX-LLaMA3-RM-v0.1
    external_lib: ${actor_rollout_ref.model.external_lib}
    use_remove_padding: False
    fsdp_config:
      min_num_params: 0
      param_offload: False
      fsdp_size: -1
  # micro_batch_size: null # will be deprecated, use micro_batch_size_per_gpu
  # micro_batch_size_per_gpu: 2 # set a number
  # max_length: null
  ulysses_sequence_parallel_size: 1 # sp size
  use_dynamic_bsz: ${critic.use_dynamic_bsz}
  forward_max_token_len_per_gpu: ${critic.forward_max_token_len_per_gpu}
  reward_manager: tinyzero

custom_reward_function:
  path: null
  name: compute_score

algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: gae
  norm_adv_by_std_in_grpo: True
  use_kl_in_reward: False
  kl_penalty: kl  # how to estimate kl divergence
  kl_ctrl:
    type: fixed
    kl_coef: 0.001
    horizon: 10000
    target_kl: 0.1

trainer:
  balance_batch: True
  total_epochs: 15
  # total_training_steps: null
  project_name: TinyZero
  experiment_name: trinity-qwen2.5-1.5b
  logger: [ 'wandb' ]
  val_generations_to_log_to_wandb: 0
  nnodes: 1
  n_gpus_per_node: 2
  save_freq: 100
  # auto: find the last ckpt to resume. If can't find, start from scratch
  resume_mode: auto # or auto or resume_path if
  resume_from_path: ""
  test_freq: 100
  critic_warmup: 0
  default_hdfs_dir: null
  remove_previous_ckpt_in_save: False
  del_local_ckpt_after_load: False
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
  val_before_train: False
  max_actor_ckpt_to_keep: 5
  max_critic_ckpt_to_keep: 5
```


- `actor_rollout_ref.model.enable_gradient_checkpointing`: Whether to enable gradient checkpointing, which will reduce GPU memory usage.
- `actor_rollout_ref.model.use_remove_padding`: Whether to remove pad tokens, which will reduce training time.
- `actor_rollout_ref.actor.use_dynamic_bsz`: Whether to reorganize the batch data, specifically to splice the shorter data to reduce the batch size in the actual training process.
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: Batch size for one GPU in one forward pass.
- `actor_rollout_ref.actor.grad_clip`: Gradient clip for actor model training.
- `actor_rollout_ref.actor.clip_ratio`: Used for compute policy loss.
- `actor_rollout_ref.actor.entropy_coeff`: Used for compute policy loss.
- `actor_rollout_ref.actor.use_kl_loss`: Whether to enable kl loss.
- `actor_rollout_ref.actor.kl_loss_coef`: The coefficient of kl loss.
- `actor_rollout_ref.actor.kl_loss_type`: How to compute kl loss, optional value is `kl`, `abs`, `mse` or `low_var_kl`.
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size`: Ulysses sequence parallel size.
- `actor_rollout_ref.actor.tau`: strength of regularization w.r.t. old / ref policy.
- `actor_rollout_ref.actor.opmd_baseline`: mean / logavgexp, applicable to opmd.
- `actor_rollout_ref.actor.use_uid`: True / False, applicable to pairwise_opmd.
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

- `algorithm`: Training algorithm settings.

- `trainer.balance_batch`: Whether to balance batch size between GPUs during training.
- `trainer.resume_mode`: Resume mode for training. Support `disable`, `auto` and `resume_path`.
- `trainer.resume_from_path`: Path to resume from.
- `trainer.critic_warmup`: The number of steps to train the critic model before actual policy learning.
- `trainer.default_hdfs_dir`: Default HDFS directory for saving checkpoints.
- `trainer.remove_previous_ckpt_in_save`: Whether to remove previous checkpoints in save.
- `trainer.del_local_ckpt_after_load`: Whether to delete local checkpoints after loading.
- `trainer.max_actor_ckpt_to_keep`: Maximum number of actor checkpoints to keep.
- `trainer.max_critic_ckpt_to_keep`: Maximum number of critic checkpoints to keep.
