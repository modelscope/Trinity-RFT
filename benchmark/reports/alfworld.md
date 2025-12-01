# ALFWorld Benchmark Results

## 1. Task Introduction

[ALFWorld](https://github.com/alfworld/alfworld) is a text-based interactive environment where agents need to complete household tasks in a virtual home environment. The agent interacts with the environment through natural language commands to accomplish tasks.

The environment is configured as follows:
* Environment: Text-based interactive environment built on TextWorld
* Action Space: Commands such as `pick`, `go to`, `place`, etc.
* Reward Structure: +1 for successfully completing the task, -0.1 otherwise
* Maximum Steps: 30 (configurable via `max_env_steps`)

See the [documentation](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_multi_turn.html) for data preparation.

## 2. Experimental Settings

We evaluate the performance of the following methods in Trinity-RFT framework with version [0.3.3](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.3.3) (verl==0.5.0, vllm==0.11.0) and compare against the latest release of rLLM with commit ID [ef6451f](https://github.com/rllm-org/rllm/commit/ef6451fbd7eba224c4a87e3fd944d7c0e2bcc0ea) (verl==0.5.0) as of Nov. 6, 2025.
Since rLLM does not support ALFWorld environment yet, we implement this task in rLLM for comparison.

In Trinity-RFT and rLLM, we respectively evaluate the performance using GRPO algorithm on this task.
We fine-tune a `Qwen2.5-3B-Instruct` model, which has been trained on a SFT dataset, on the training tasks with GRPO and other methods. For all methods, we fix key parameters to `batch_size=32`, `repeat_times=8`, `lr=1e-6`, and `kl_coef=0.001`.

For better efficiency, we use 64 rollout workers in rLLM and set the `explorer.engine_num` to 4 and `explorer.runner_per_model` to 8 in Trinity-RFT.

## 3. Results and Analysis

We compare the sample efficiency of different methods by plotting the reward and test score vs. training steps. As shown in the following figures, Trinity-RFT and rLLM reach similar training and test results at the same step.

![](../../docs/sphinx_doc/assets/bench_alfworld_step.png)

We further compare the efficiency on the ALFWorld task.
The following table details the wall-clock time required for each method to reach the specific performance thresholds, i.e., reward = 0.8 and test score = 0.6.

| Method | Training Reward | Time to Reach Target (Hours) | Speedup |
|----------|------------------|-------------------------------|---------|
| rLLM | 0.830 | 9.33 | - |
| Trinity-RFT | 0.826 | 2.53 | 3.69× |


| Method | Test Score | Time to Reach Target (Hours) | Speedup |
|----------|------------------|-------------------------------|---------|
| rLLM | 0.670 | 6.65 | - |
| Trinity-RFT | 0.632 | 1.14 | 5.83× |

The results show that the Trinity-RFT achieves a noticeable speedup on the ALFWorld task, also shown in the following figures.
The primary reason for the efficiency lies in the difference between the rollout mechanisms of Trinity-RFT and rLLM. Trinity-RFT uses multiprocessing during rollout, whereas rLLM employs multithreading, which restricts the parallelism of the rollout process in ALFWorld environment given that this environment is not thread-safe (refer to [this issue](https://github.com/alfworld/alfworld/issues/71)).

![](../../docs/sphinx_doc/assets/bench_alfworld_time.png)
