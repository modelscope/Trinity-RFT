(Installation)=
# Installation

For installing Trinity-RFT, you have three options: from source (recommended), via PyPI, or using Docker.

Before installing, ensure your system meets the following requirements:

- **Python**: Version 3.10 to 3.12 (inclusive)
- **CUDA**: Version 12.4 to 12.8 (inclusive)
- **GPUs**: At least 2 GPUs

---

## From Source (Recommended)

This method is best if you plan to customize or contribute to Trinity-RFT.

### 1. Clone the Repository

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT
```

### 2. Set Up a Virtual Environment

Choose one of the following options:

#### Using Conda

```bash
conda create -n trinity python=3.10
conda activate trinity

pip install -e ".[dev]"
pip install -e ".[flash_attn]"
# if you encounter issues when installing flash-attn, try:
# pip install flash-attn==2.8.1 --no-build-isolation
```

#### Using venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pip install -e ".[flash_attn]"
# if you encounter issues when installing flash-attn, try:
# pip install flash-attn==2.8.1 --no-build-isolation
```

#### Using `uv`

[`uv`](https://github.com/astral-sh/uv) is a modern Python package installer.

```bash
uv sync --extra dev --extra flash_attn
```

---

## Via PyPI

If you just want to use the package without modifying the code:

```bash
pip install trinity-rft==0.3.1
pip install flash-attn==2.8.1
```

Or with `uv`:

```bash
uv pip install trinity-rft==0.3.1
uv pip install flash-attn==2.8.1
```

---

## Using Docker

We provide a Docker setup for hassle-free environment configuration.

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# Build the Docker image
## Tip: You can modify the Dockerfile to add mirrors or set API keys
docker build -f scripts/docker/Dockerfile -t trinity-rft:latest .

# Run the container, replacing <path_to_your_data_and_checkpoints> with your actual path
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v <path_to_your_data_and_checkpoints>:/data \
  trinity-rft:latest
```

```{note}
For training with **Megatron-LM**, please refer to {ref}`Megatron-LM Backend <Megatron-LM>`.
```

---

## Troubleshooting

If you encounter installation issues, refer to the FAQ or [GitHub Issues](https://github.com/modelscope/Trinity-RFT/issues).
