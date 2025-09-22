# GRPO with LoRA

This example shows the usage of LoRA on the GSM8K dataset.

## GRPO training
Compared with full model fine-tuning, Trinity-RFT enable LoRA by providing the `lora_configs` field as follows:

```yaml
project: "Trinity-RFT-gsm8k"
name: "qwen2.5-1.5B-gsm8k"
model:
  lora_configs:
  - name: lora
    lora_rank: 16
    lora_alpha: 16
synchronizer:
  sync_method: 'checkpoint'
```

Note that the `lora_rank` and `lora_alpha` are hyperparameters that need to be tuned. For `lora_rank`, a very small value can lead to slower convergence or worse training performance, while a very large value can lead to memory and performance issues.

For now, we only support a single-lora training and synchronizing via `checkpoint`.

## Benchmark with LoRA
After training, we can evaluate the performance of checkpoints via the `bench` mode. Some key configurations are shown below:

```yaml
mode: bench
project: "Trinity-RFT-gsm8k"  # same as training
name: "qwen2.5-1.5B-gsm8k"  # same as training
model:
  lora_configs:  # same as training
  - name: lora
    lora_rank: 16
    lora_alpha: 16
explorer:
  rollout_model:
    engine_num: 2  # ensure all gpus are used for benchmarking
    tensor_parallel_size: 4
synchronizer:
  sync_method: 'checkpoint'
```
