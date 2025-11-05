# 快速上手

本教程介绍了使用 Trinity-RFT 运行 RFT 的快速入门指南。

## 第 0 步：环境准备

请按照[安装指南](./trinity_installation.md)中的说明进行环境设置。


## 第 1 步：模型和数据准备


**模型准备**

将 Qwen2.5-1.5B-Instruct 模型下载到本地目录 `$MODEL_PATH/Qwen2.5-1.5B-Instruct`：

```bash
# 使用 Modelscope
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir $MODEL_PATH/Qwen2.5-1.5B-Instruct

# 使用 Huggingface
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $MODEL_PATH/Qwen2.5-1.5B-Instruct
```

更多关于模型下载的细节请参考 [ModelScope](https://modelscope.cn/docs/models/download) 或 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)。

**数据准备**

将 GSM8K 数据集下载到本地目录 `$DATASET_PATH/gsm8k`：

```bash
# 使用 Modelscope
modelscope download --dataset modelscope/gsm8k --local_dir $DATASET_PATH/gsm8k

# 使用 Huggingface
huggingface-cli download openai/gsm8k --repo-type dataset --local-dir $DATASET_PATH/gsm8k
```

更多关于数据集下载的细节请参考 [ModelScope](https://modelscope.cn/docs/datasets/download) 或 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space)。

## 第 2 步：配置实验并运行

### Trinity-RFT 的同步模式

我们在同步模式下运行实验，其中 Explorer 和 Trainer 轮流执行。要启用此模式，需将 `mode` 设置为 `both`（默认）并合理设置 `sync_interval`。较小的 `sync_interval` 值使训练更接近 on-policy 设置。例如，我们将 `sync_interval` 设为 1 来模拟 on-policy 场景。

### 使用 GRPO 算法

本实验使用 [`gsm8k.yaml`](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k/gsm8k.yaml) 中的配置。以下是 `gsm8k.yaml` 中一些重要配置项：

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
      path: ${oc.env:TRINITY_TASKSET_PATH,openai/gsm8k}
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
      path: ${oc.env:TRINITY_TASKSET_PATH,openai/gsm8k}
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


### 运行实验

使用以下命令启动 RFT 流程：

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```



## 进阶选项：带 SFT warmup 的 RFT

在进行 RFT 之前，我们可以先使用 SFT 作为预热步骤。Trinity-RFT 支持通过在配置文件中设置 `stages` 来添加 SFT 预热阶段。`experience_buffer` 指定用于 SFT warmup 的数据集，`total_steps` 指定 SFT warmup 的训练步数。

```yaml
# 在 gsm8k.yaml 中正确添加以下配置
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
  - stage_name: rft  # 留空则使用原有的 RFT 配置
```

以下命令将按顺序运行 SFT 和 RFT：

```bash
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```
