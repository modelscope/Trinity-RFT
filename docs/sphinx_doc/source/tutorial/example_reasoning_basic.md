# Quick Start

This tutorial shows a quick start guide for running RFT with Trinity-RFT.

## Step 0: Environment Preparation

Please follow the instructions in [Installation](./trinity_installation.md) to set up the environment.

## Step 1: Model and Data Preparation


**Model Preparation.**

Download the Qwen2.5-1.5B-Instruct model to the local directory `$MODEL_PATH/Qwen2.5-1.5B-Instruct`:

```bash
# Using Modelscope
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir $MODEL_PATH/Qwen2.5-1.5B-Instruct

# Using Huggingface
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $MODEL_PATH/Qwen2.5-1.5B-Instruct
```

More details on model downloading are referred to [ModelScope](https://modelscope.cn/docs/models/download) or [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli).

**Data Preparation.**

Download the GSM8K dataset to the local directory `$DATASET_PATH/gsm8k`:

```bash
# Using Modelscope
modelscope download --dataset modelscope/gsm8k --local_dir $DATASET_PATH/gsm8k

# Using Huggingface
huggingface-cli download openai/gsm8k --repo-type dataset --local-dir $DATASET_PATH/gsm8k
```

More details on dataset downloading are referred to [ModelScope](https://modelscope.cn/docs/datasets/download) or [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space).

## Step 2: Set up Configuration and Run Experiment

### Synchronous Mode of Trinity-RFT

We run the experiment in a synchronous mode where the Explorer and Trainer operate in turn. To enable this mode, we config `mode` to `both` (default) and set `sync_interval` properly. A smaller value of `sync_interval` makes the training closer to an on-policy setup. For example, we set `sync_interval` to 1 to simulate an on-policy setup.

### Use GRPO Algorithm

We use the configurations in [`gsm8k.yaml`](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k/gsm8k.yaml) for this experiment. Some important setups of `gsm8k.yaml` are listed in the following:


```yaml
project: <project_name>
name: <experiment_name>
checkpoint_root_dir: ${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}
algorithm:
  algorithm_type: grpo
  repeat_times: 8
  optimizer:
    lr: 1e-5
model:
  model_path: ${oc.env:TRINITY_MODEL_PATH,Qwen/Qwen2.5-1.5B-Instruct}
cluster:
  node_num: 1
  gpu_per_node: 2
buffer:
  total_epochs: 1
  batch_size: 128
  explorer_input:
    taskset:
      name: gsm8k
      storage_type: file
      path: 'openai/gsm8k'
      subset_name: 'main'
      split: 'train'
      format:
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 1.0
      default_workflow_type: 'math_workflow'
    eval_tasksets:
    - name: gsm8k-eval
      storage_type: file
      path: 'openai/gsm8k'
      subset_name: 'main'
      split: 'test'
      format:
        prompt_key: 'question'
        response_key: 'answer'
      default_workflow_type: 'math_workflow'
  trainer_input:
    experience_buffer:
      name: gsm8k_buffer
      storage_type: queue
      path: 'sqlite:///gsm8k.db'
explorer:
  eval_interval: 50
  runner_per_model: 16
  rollout_model:
    engine_num: 1
synchronizer:
  sync_method: 'nccl'
  sync_interval: 1
trainer:
  save_interval: 100
```


### Run the Experiment

Run the RFT process with the following command:

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```



## Optional: RFT with SFT Warmup

Before RFT, we may use SFT as a warmup step. Trinity-RFT supports adding SFT warmup stage before RFT by setting `stages` in the config file. The `experience_buffer` specifies the dataset used for SFT warmup, and `total_steps` specifies the number of training steps for SFT warmup.

```yaml
# Properly add the following configs in gsm8k.yaml
stages:
  - stage_name: sft_warmup
    mode: train
    algorithm:
      algorithm_type: sft
    buffer:
      train_batch_size: 128
      total_steps: 10
      trainer_input:
        experience_buffer:
          name: sft_warmup_dataset
          path: /PATH/TO/YOUR/SFT/DATASET
  - stage_name: rft  # leave empty to use the original configs for RFT
```

The following command runs SFT and RFT in sequence:

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```
