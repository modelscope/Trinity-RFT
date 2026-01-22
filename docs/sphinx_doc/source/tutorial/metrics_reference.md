# Metrics Reference

This document provides an overview of the metric categories used in Trinity-RFT for tracking performance.

## Metric Naming Convention

Most metrics follow a hierarchical naming convention: `{category}/{taskset_name}/{metric_name}/{statistic}`

- **Category**: Broad functional area (rollout, eval, time, actor, critic, etc.)
- **Taskset name**: Name of the taskset used, only applicable for eval metrics
- **Metric name**: Specific metric being measured
- **Statistic**: Aggregation method (mean, max, min, std, etc.) if applicable


## Metric Categories

In the following, metrics are categorized by their source component (where they are generated) and their metric prefix (the first part of the metric name).

### Explorer Metrics

Explorer metrics track performance during the rollout phase where the model generates responses, including rollout metrics (`rollout/`), eval metrics (`eval/`), and some time metrics (`time/`).

#### Metric Aggregation Levels

Consider a task with `repeat_times` runs, an exploration step with `batch_size` tasks, and an evalutation step with `eval_taskset_size` tasks. Explorer metrics are computed and aggregated at different levels:

- **Task level**: Metrics aggregated across `repeat_times` runs of the same task. For exploration tasks, the metrics are aggregated across all runs of the task, e.g., `rollout/accuracy` is the average accuracy of all runs of the task. For evaluation tasks, task-level metrics include (e.g., `mean@4`, `std@4`, `best@2`, `worst@2`) that are computed from k runs of the task.

- **Step level**: For most cases, the metrics are reported at the step level. For example, `rollout/accuracy/mean`, `rollout/accuracy/max`, `rollout/accuracy/min` are the average, max, and min accuracy (`rollout/accuracy`) of all tasks in the step. As for evaluation tasks, we report the mean of the metric across all evaluation tasks by default; if you want to return detailed statistics, you can set `monitor.detailed_stats` to `True` in the config.


#### Rollout Metrics (`rollout/`)

Rollout metrics track performance during the rollout phase where the model generates responses.

- **Format**: `rollout/{metric_name}/{statistic}`
- **Examples**:
  - `rollout/accuracy/mean`: Average accuracy of generated responses
  - `rollout/format_score/std`: Average format correctness score
  - `rollout/finished_task_count`: Number of completed rollout tasks
  - `rollout/model_version`: Model version used for rollout
  - `rollout/time/run_execution/mean`: Average execution time per rollout


#### Eval Metrics (`eval/`) and Benchmark Metrics (`bench/`)

Evaluation metrics measure model performance on held-out evaluation tasks. These metrics are computed during periodic evaluation runs.

- **Format**: `eval/{task_name}/{metric_name}/{statistic}` or `bench/{task_name}/{metric_name}/{statistic}`
- **Examples**:
  - `eval/gsm8k-eval/accuracy/mean@4`: Mean accuracy across repeat_times=4 runs
  - `eval/gsm8k-eval/accuracy/best@2`: Best accuracy value across k=2 runs, computed by bootstrap method
  - `eval/gsm8k-eval/accuracy/worst@2`: Worst accuracy value across k=2 runs, computed by bootstrap method
  - `bench/gsm8k-eval/accuracy/mean@4`: Mean accuracy across repeat_times=4 runs

- **Note**:
  - Eval and bench metrics are computed in the same way, the only difference is the prefix of the metric name.
  - By default, only the *mean* of the metric is returned. If you want to return detailed statistics, you can set `monitor.detailed_stats` to `True` in the config.


#### Time Metrics (`time/`)

Time metrics measure execution duration for various operations throughout the training pipeline.

- **Format**: `time/{operation_name}`
- **Examples**:
  - `time/eval`: Time from the start of submitting evaluation tasks to the end of the evaluation phase; this duration includes both evaluation tasks and some rollout tasks.
  - `time/read_experience`: Time to read experiences from taskset
  - `time/wait_explore_step`: Time waiting for a rollout/exploration step completion
  - `time/update_critic`: Time to update critic model
  - `time/update_actor`: Time to update actor model
  - `time/sync_weight`: Time to synchronize model weights
  - `time/save_checkpoint`: Time to save model checkpoint
  - `time/train_step`: Total time for one training step
  - `time/trainer_sync_interval`: Time interval between trainer synchronizations

**Note**:
  - Time measuring can be inaccurate due to the asynchronous nature of the exploration pipeline, but it is still useful for monitoring the overall training progress.
  - Above metrics are reported in seconds unless otherwise specified.
  - Some training operations also report per-token timing metrics with the prefix `timing_per_token_ms/` (e.g., `timing_per_token_ms/update_actor`, `timing_per_token_ms/update_critic`, `timing_per_token_ms/adv`, `timing_per_token_ms/values`). These metrics normalize execution time by the number of tokens processed, providing efficiency measurements independent of batch size.


### Training Metrics 

This category includes metrics that track the training dynamics of the policy (actor) model (`actor/`) and the value function (critic) model (`critic/`), as well as some performance metrics (`perf/`, `global_seqlen/`, `response_length/`, `prompt_length/`, `time/`).

#### Actor Metrics (`actor/`)

Actor metrics track the training dynamics of the policy (actor) model in reinforcement learning.

- **Format**: `actor/{metric_name}`
- **Examples**:
  - `actor/pg_loss`: Policy gradient loss
  - `actor/entropy_loss`: Entropy regularization loss
  - `actor/kl_loss`: KL divergence loss
  - `actor/ppo_kl`: PPO-specific KL divergence
  - `actor/pg_clipfrac`: Fraction of policy gradient updates clipped
  - `actor/final_loss`: Final loss used to update the actor model, usually a combination of policy gradient loss, entropy regularization loss, and KL divergence loss.

#### Critic Metrics (`critic/`)

Critic metrics track the training dynamics of the value function (critic) model.

- **Format**: `critic/{metric_name}/{statistic}`
- **Examples**:
  - `critic/score/mean`: Mean sequence-level score
  - `critic/rewards/mean`: Mean sequence-level reward
  - `critic/advantages/mean`: Mean advantage values
  - `critic/returns/mean`: Mean return values

#### Performance Metrics (`perf/`)

Performance metrics measure computational efficiency and resource utilization.

- **Format**: `perf/{metric_name}`
- **Examples**:
  - `perf/mfu/actor`: Model FLOPs Utilization (MFU) for actor
  - `perf/mfu/critic`: Model FLOPs Utilization (MFU) for critic
  - `perf/mfu/actor_infer`: Model FLOPs Utilization for actor inference (when recomputing logprobs)
  - `perf/max_memory_allocated_gb`: Peak GPU memory allocated
  - `perf/max_memory_reserved_gb`: Peak GPU memory reserved
  - `perf/cpu_memory_used_gb`: CPU memory usage
  - `perf/total_num_tokens`: Total number of tokens processed
  - `perf/time_per_step`: Time per training step
  - `perf/throughput`: Tokens processed per second

#### Global Sequence Length Metrics (`global_seqlen/`)

Global sequence length metrics track sequence length statistics across the training batch.

- **Format**: `global_seqlen/{statistic}`
- **Examples**:
  - `global_seqlen/mean`: Mean sequence length
  - `global_seqlen/min`: Minimum sequence length
  - `global_seqlen/max`: Maximum sequence length
  - `global_seqlen/minmax_diff`: Difference between max and min
  - `global_seqlen/balanced_min`: Balanced minimum (for load balancing)
  - `global_seqlen/balanced_max`: Balanced maximum (for load balancing)

#### Response and Prompt Length Metrics (`response_length/` and `prompt_length/`)

Metrics tracking the length of generated responses and input prompts.

- **Format**: `response_length/{statistic}` or `prompt_length/{statistic}`
- **Examples**:
  - `response_length/mean`: Mean response length in tokens
  - `response_length/max`: Maximum response length
  - `response_length/min`: Minimum response length
  - `response_length/clip_ratio`: Fraction of responses clipped to max length
  - `prompt_length/mean`: Mean prompt length in tokens
  - `prompt_length/clip_ratio`: Fraction of prompts clipped to max length


**Note**:
  - `/clip_ratio` means the fraction of responses/prompts that matches the max length (instead of being truncated).


### Data Processing Metrics

This category includes metrics that track the processing of experiences through various pipeline operators (`experience_pipeline/`) and data sampling statistics (`sample/`).

#### Experience Pipeline Metrics (`experience_pipeline/` and `time/experience_pipeline/`)

Experience pipeline metrics track the processing of experiences through various pipeline operators. Each metric represents the count of the specific operator in one step.

- **Format**: `experience_pipeline/{metric_name}`
- **Examples**:
  - `experience_pipeline/experience_count`: Number of experiences processed
  - `experience_pipeline/filtered_count`: Number of experiences filtered out
  - `experience_pipeline/group_advantages/reward_mean/mean`: Mean reward statistics
  - `time/experience_pipeline/operator/{operator_name}`: Time for specific pipeline operators
  - `time/experience_pipeline/write`: Time to write experiences to storage
  - `time/experience_pipeline/total`: Total time for experience processing

#### Sample Metrics (`sample/`)

Sample metrics track data sampling statistics during training.

- **Format**: `sample/{metric_name}`
- **Examples**:
  - `sample/model_version/mean`: Mean model version of sampled experiences
  - `sample/task_count`: Number of tasks in the sampled batch
