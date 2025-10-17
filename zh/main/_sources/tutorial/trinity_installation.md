(Installation)=
# 安装指南

安装 Trinity-RFT 有三种方式：源码安装（推荐）、通过 PyPI 安装，或使用 Docker。

在安装前，请确保您的系统满足以下要求：

- **Python**：3.10 至 3.12（包含）
- **CUDA**：12.4 至 12.8（包含）
- **GPU**：至少 2 块 GPU

---

## 源码安装（推荐）

如需修改、扩展 Trinity-RFT，推荐使用此方法。

### 1. 克隆仓库

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT
```

### 2. 创建虚拟环境

可选择以下任一方式：

#### 使用 Conda

```bash
conda create -n trinity python=3.10
conda activate trinity

pip install -e ".[dev]"
pip install -e ".[flash_attn]"
# 如果安装 flash-attn 时遇到问题，可尝试：
# pip install flash-attn==2.8.1 --no-build-isolation
```

#### 使用 venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pip install -e ".[flash_attn]"
# 如果安装 flash-attn 时遇到问题，可尝试：
# pip install flash-attn==2.8.1 --no-build-isolation
```

#### 使用 `uv`

[`uv`](https://github.com/astral-sh/uv) 是现代的 Python 包管理工具。

```bash
uv sync --extra dev --extra flash_attn
```

---

## 通过 PyPI 安装

如果您只需使用 Trinity-RFT 而不打算修改代码：

```bash
pip install trinity-rft==0.3.1
pip install flash-attn==2.8.1
```

或使用 `uv`：

```bash
uv pip install trinity-rft==0.3.1
uv pip install flash-attn==2.8.1
```

---

## 使用 Docker

我们提供了 Docker 环境，方便快速配置。

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# 构建 Docker 镜像
## 提示：可根据需要修改 Dockerfile 添加镜像源或设置 API 密钥
docker build -f scripts/docker/Dockerfile -t trinity-rft:latest .

# 运行容器，请将 <path_to_your_data_and_checkpoints> 替换为实际需要挂载的路径
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v <path_to_your_data_and_checkpoints>:/data \
  trinity-rft:latest
```

```{note}
如需使用 **Megatron-LM** 进行训练，请参考 {ref}`Megatron-LM Backend <Megatron-LM>`。
```

---

## 常见问题

如遇安装问题，请参考 FAQ 或 [GitHub Issues](https://github.com/modelscope/Trinity-RFT/issues)。
