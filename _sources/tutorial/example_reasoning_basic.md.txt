# A quick example with GSM8k

This example shows how to run RFT with the Qwen-2.5-1.5B-Instruct model and GSM8K dataset.

## Step 1: Model and Data Preparation


**Model Preparation.**

Download the Qwen-2.5-1.5B-Instruct model to the local directory `$MODEL_PATH/Qwen2.5-1.5B-Instruct`:

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

We run the experiment in a synchronous mode where the Explorer and Trainer operate in turn. To enable this mode, we config `mode` to `both` (default) and set `sync_iteration_interval` properly. A smaller value of `sync_iteration_interval` makes the training closer to an on-policy setup.

```yaml
mode: both
synchronizer:
  sync_method: 'nccl'
  sync_iteration_interval: 2
```

### Use GRPO or PPO Algorithm

We use the configurations in [`gsm8k.yaml`](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k/gsm8k.yaml) and [`train_gsm8k.yaml`](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k/train_gsm8k.yaml) for this experiment. Some important setups are listed in the following:


```yaml
# In gsm8k.yaml
explorer:
  repeat_times: {number of rollouts for each task}

# In train_gsm8k.yaml
actor_rollout_ref:
  actor:
    use_kl_loss: True (fro GRPO) / False (for PPO)
    kl_loss_coef: 0.001
algorithm:
  adv_estimator: grpo (fro GRPO) / gae (for PPO)
```

### Run the Experiment

Run the RFT process with the following command:
```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```



## Optional: RFT with SFT Warmup

Before RFT, we may use SFT as a warmup step. We need to set `trainer.sft_warmup_iteration > 0` and prepare the SFT data to `buffer.train_dataset.path=$DATASET_PATH/{sft_data}`.

```yaml
# Properly set the following configs in gsm8k.yaml
buffer:
  sft_warmup_dataset:
    storage_type: file
    algorithm_type: sft
    path: <$DATASET_PATH/{sft_data}>
    kwargs:
      prompt_type: <prompt_type> # messages/plaintext/chatpair
      prompt_key: <prompt_key>
      response_key: <response_key>
trainer:
  sft_warmup_iteration: 10
```

The following command runs SFT and RFT in sequence:
```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```
