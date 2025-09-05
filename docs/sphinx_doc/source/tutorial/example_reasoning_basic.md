# Quick Start

This tutorial shows a quick start guide for running RFT with Trinity-RFT.

## Step 0: Environment Preparation

Minimal environment requirements:

- GPUs: At least 2 GPUs
- CUDA: Version >= 12.4
- Python: Version >= 3.10

```shell
# Pull the source code from GitHub
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# Create a new environment using Conda or venv
# Option 1: Conda
conda create -n trinity python=3.10
conda activate trinity

# Option 2: venv
python3.10 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode
# for bash
pip install -e .[dev]
# for zsh
pip install -e .\[dev\]

# Install flash-attn after all dependencies are installed
# Note: flash-attn will take a long time to compile, please be patient.
pip install flash-attn -v
# Try the following command if you encounter errors during installation
# pip install flash-attn -v --no-build-isolation
```

Installation using pip:

```shell
pip install trinity-rft
```

Installation from docker:

We provided a dockerfile for Trinity-RFT.

```shell
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# build the docker image
# Note: you can edit the dockerfile to customize the environment
# e.g., use pip mirrors or set api key
docker build -f scripts/docker/Dockerfile -t trinity-rft:latest .

# run the docker image
docker run -it --gpus all --shm-size="64g" --rm -v $PWD:/workspace -v <root_path_of_data_and_checkpoints>:/data trinity-rft:latest
```


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
checkpoint_root_dir: /PATH/TO/CHECKPOINT/
algorithm:
  algorithm_type: grpo
  repeat_times: 8
model:
  model_path: /PATH/TO/MODEL/
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
      path: <$DATASET_PATH/gsm8k>
      subset_name: 'main'
      split: 'train'
      format:
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 1.0
    eval_tasksets:
    - name: gsm8k-eval
      storage_type: file
      path: <$DATASET_PATH/gsm8k>
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
  runner_num: 16
  rollout_model:
    engine_type: vllm_async
    engine_num: 1
synchronizer:
  sync_method: 'nccl'
  sync_interval: 1
trainer:
  save_interval: 100
  trainer_config:
    actor_rollout_ref:
      actor:
        optim:
          lr: 1e-5
```


### Run the Experiment

Run the RFT process with the following command:
```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```



## Optional: RFT with SFT Warmup

Before RFT, we may use SFT as a warmup step. We need to set `buffer.trainer_input.sft_warmup_steps > 0` and prepare the SFT data to `buffer.trainer_input.sft_warmup_dataset.path=$DATASET_PATH/{sft_data}`.

```yaml
# Properly add the following configs in gsm8k.yaml
buffer:
  trainer_input:
    sft_warmup_dataset:
      storage_type: file
      path: <$DATASET_PATH/{sft_data}>
      format:
        prompt_type: <prompt_type> # messages/plaintext
        prompt_key: <prompt_key>
        response_key: <response_key>
    sft_warmup_steps: 10
```

The following command runs SFT and RFT in sequence:
```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```
