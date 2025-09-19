# GRPO with LoRA

This example shows the usage of GRPO with LoRA on the GSM8K dataset.

Compared with full model fine-tuning, Trinity-RFT enable LoRA by providing the `lora_configs` field as follows:

```yaml
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
