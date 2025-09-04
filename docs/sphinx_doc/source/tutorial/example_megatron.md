# Megatron-LM Backend

This guide walks you through how to train models using **Megatron-LM** in a clear way.

---

## Step 1: Installation

### Minimum Requirements

Before you begin, make sure your system meets these requirements:

- **GPUs**: At least 2 GPUs (for distributed training)
- **CUDA**: Version 12.4 or higher
- **Python**: Version 3.10 or higher

---

### Install Dependencies

Start by cloning the repository and setting up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT
```

#### Option A: Using Conda

```bash
# Create and activate a new environment
conda create -n trinity python=3.10
conda activate trinity
```

#### Option B: Using venv

```bash
# Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
```

#### Install the Package

Install the project in editable mode with Megatron support:

```bash
# For bash users
pip install -e .[megatron]

# For zsh users (escape the brackets)
pip install -e .\[megatron\]
```

#### Install Flash Attention

After installing the base dependencies, install `flash-attn`. This may take several minutes to compile — please be patient!

```bash
pip install flash-attn==2.8.1 -v
```

If you run into installation issues, try this alternative:

```bash
pip install flash-attn -v --no-build-isolation
```

#### Install Apex (from NVIDIA)

Finally, install NVIDIA's Apex library for mixed-precision training:

```bash
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    --resume-retries 999 git+https://github.com/NVIDIA/apex.git
```

---

### Alternative: Use Docker

We provide a Docker setup to simplify environment management.

#### Build the Docker Image

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# Build the image
docker build -f scripts/docker_for_megatron/Dockerfile -t trinity-rft-megatron:latest .
```

> 💡 You can customize the Dockerfile before building — for example, to add pip mirrors or set API keys.

#### Run the Container

```bash
docker run -it \
    --gpus all \
    --shm-size="64g" \
    --rm \
    -v $PWD:/workspace \
    -v <your_data_and_checkpoints_path>:/data \
    trinity-rft-megatron:latest
```

Replace `<your_data_and_checkpoints_path>` with the actual path on your machine where datasets and model checkpoints are stored.

---

## Step 2: Configure and Run Training

Most configuration settings are covered in the [Quick Start Guide](../../docs/sphinx_doc/source/tutorial/example_reasoning_basic.md#step-0-environment-preparation). Here, we'll focus only on **Megatron-LM-specific** settings.

### Megatron Configuration Example

Below is an example of how to configure the actor, reference model, and critic to use Megatron-LM:

```yaml
actor_rollout_ref:
  ...
  actor:
    strategy: megatron  # Kept for backward compatibility
    megatron:
      # Model parallelism settings
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      expert_model_parallel_size: 1

      # Offloading (set to false unless you're memory-constrained)
      param_offload: false
      grad_offload: false
      optimizer_offload: false

      # Use mBridge for parameter import/export (optional)
      use_mbridge: false

      # Use Megatron checkpoint
      use_dist_checkpointing: false
      dist_checkpointing_path: null

      # Recomputation settings (helps save memory during training)
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

### Training Mixture-of-Experts (MoE) Models

If you're training an MoE model like **Qwen/Qwen3-30B-A3B**, you’ll need to take one of the following two approaches to ensure it works properly:

1. **Use MBridge (Recommended)**:
   Simply set `use_mbridge: true` in your configuration file. This enables the necessary support for MoE models directly.

2. **Convert the model manually**:
   If you prefer not to use MBridge, set `use_mbridge: false`. Before training, you must first convert your Hugging Face model to the MCore format using the [Hugging Face to MCore converter](https://github.com/volcengine/verl/blob/main/scripts/converter_hf_to_mcore.py) from the **verl** repository. After conversion, update your config with:
   - `use_dist_checkpointing: true`
   - `dist_checkpointing_path: /PATH/TO/CONVERTED/MODEL/`

> ⚠️ Important: If you skip both steps, the MoE model may fail to load or train correctly. Make sure to follow one of the two options above.
