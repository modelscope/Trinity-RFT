# Megatron-LM 支持

本指南将清晰地引导你如何使用 **Megatron-LM** 来训练模型。

---

## 步骤 1：安装

### 最低要求

在开始之前，请确保你的系统满足以下要求：

- **GPU**：至少 2 块 GPU（用于分布式训练）
- **CUDA**：版本 12.4 或更高
- **Python**：版本 3.10 或更高

---

### 安装依赖项

首先克隆仓库并创建虚拟环境：

```bash
# 克隆仓库
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT
```

#### 选项 A：使用 Conda

```bash
# 创建并激活新环境
conda create -n trinity python=3.10
conda activate trinity
```

#### 选项 B：使用 venv

```bash
# 创建并激活虚拟环境
python3.10 -m venv .venv
source .venv/bin/activate
```

#### 安装包

以可编辑模式安装项目，并启用 Megatron 支持：

```bash
# 针对 bash 用户
pip install -e .[megatron]

# 针对 zsh 用户（需转义括号）
pip install -e .\[megatron\]
```

#### 安装 Flash Attention

安装基础依赖后，安装 `flash-attn`。编译过程可能需要几分钟，请耐心等待。

```bash
pip install flash-attn==2.8.1 -v
```

如果遇到安装问题，可尝试以下替代命令：

```bash
pip install flash-attn -v --no-build-isolation
```

#### 安装 Apex（来自 NVIDIA）

最后，安装 NVIDIA 的 Apex 库以支持混合精度训练：

```bash
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    --resume-retries 999 git+https://github.com/NVIDIA/apex.git
```

---

### 替代方案：使用 Docker

我们提供了 Docker 配置以简化环境管理。

#### 构建 Docker 镜像

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# 构建镜像
docker build -f scripts/docker_for_megatron/Dockerfile -t trinity-rft-megatron:latest .
```

> 💡 你可以在构建前自定义 Dockerfile —— 例如添加 pip 镜像源或设置 API 密钥。

#### 运行容器

```bash
docker run -it \
    --gpus all \
    --shm-size="64g" \
    --rm \
    -v $PWD:/workspace \
    -v <your_data_and_checkpoints_path>:/data \
    trinity-rft-megatron:latest
```

请将 `<your_data_and_checkpoints_path>` 替换为你机器上存储数据集和模型检查点的实际路径。

---

## 步骤 2：配置并运行训练

大多数配置设置已在 [快速入门指南](./example_reasoning_basic.md) 中涵盖。此处我们仅关注 **Megatron-LM 特有** 的配置。

### Megatron 配置示例

以下是将 actor、reference model 和 critic 配置为使用 Megatron-LM 的示例：

```yaml
actor_rollout_ref:
  ...
  actor:
    strategy: megatron  # 为保持向后兼容性保留
    megatron:
      # 模型并行设置
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1

      # offload 设置（除非内存受限，否则设为 false）
      param_offload: false
      grad_offload: false
      optimizer_offload: false

      # 使用 mBridge 进行参数导入/导出（可选）
      use_mbridge: false

      # 使用 Megatron checkpointing
      use_dist_checkpointing: false
      dist_checkpointing_path: null

      # 重计算设置（有助于训练期间节省内存）
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
  ...
  ref:
    megatron:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1
      param_offload: false
      grad_offload: false
      optimizer_offload: false
      use_mbridge: false
      use_dist_checkpointing: false
      dist_checkpointing_path: null
      override_transformer_config:
        recompute_granularity: full
        recompute_method: uniform
        recompute_num_layers: 1
  ...

critic:
  strategy: megatron
  megatron:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1
    expert_model_parallel_size: 1
    param_offload: false
    grad_offload: false
    optimizer_offload: false
    use_mbridge: false
    use_dist_checkpointing: false
    dist_checkpointing_path: null
    override_transformer_config:
      recompute_granularity: full
      recompute_method: uniform
      recompute_num_layers: 1
  ...
```

---

### 训练 Mixture-of-Experts (MoE) 模型

如果你正在训练像 **Qwen/Qwen3-30B-A3B** 这样的 MoE 模型，则需要采用以下两种方法之一，以确保其正常工作：

1. **使用 MBridge（推荐）**：
   只需在配置文件中设置 `use_mbridge: true`。这将直接启用对 MoE 模型所需的支持。

2. **手动转换模型**：
   如果你不希望使用 MBridge，请设置 `use_mbridge: false`。在训练前，你必须先使用 **verl** 仓库中的 [Hugging Face 到 MCore 转换器](https://github.com/volcengine/verl/blob/main/scripts/converter_hf_to_mcore.py) 将 Hugging Face 模型转换为 MCore 格式。转换完成后，在配置中更新：
   - `use_dist_checkpointing: true`
   - `dist_checkpointing_path: /PATH/TO/CONVERTED/MODEL/`

> ⚠️ 重要提示：如果跳过上述任一方法，MoE 模型可能无法正确加载或训练。请务必选择以上两种方式之一。
