# Trinity with Tinker Backend

This example demonstrates how to use Trinity with the [Tinker](https://thinkingmachines.ai/tinker/) backend, which enables model training on devices without GPUs.

## Setup Instructions

### 1. API Key Configuration
Before starting Ray, you must set the `TRINITY_API_KEY` environment variable to your Tinker API key to enable proper access to Tinker's API:

```bash
export TRINITY_API_KEY=your_tinker_api_key
```

### 2. Configuration File
Configure the Tinker backend in your YAML configuration file by setting the `model.tinker` parameters as shown below:

```yaml
model:
  tinker:
    enable: true
    base_model: null
    rank: 32
    seed: null
    train_mlp: true
    train_attn: true
    train_unembed: true
```

### 3. Configuration Parameters Explained

- **`tinker`**: Optional Tinker-specific configuration section. **Important**: When Tinker is enabled, any LoRA configuration settings will be ignored.
  - **`enable`**: Whether to activate the Tinker backend. Default: `false`
  - **`base_model`**: Path to the base model for Tinker. If not specified (`null`), it defaults to the `model_path` defined elsewhere in your config
  - **`rank`**: The LoRA rank that controls the size of the adaptation matrices. Default: `32`
  - **`seed`**: Random seed for reproducible Tinker operations. If not specified (`null`), no specific seed is set
  - **`train_mlp`**: Whether to train the MLP (feed-forward) layers. Default: `true`
  - **`train_attn`**: Whether to train the attention layers. Default: `true`
  - **`train_unembed`**: Whether to train the unembedding (output) layer. Default: `true`

## Usage Notes

Once configured, Trinity works with the Tinker backend just like it does with the standard veRL training backend, with two important limitations:
1. **Entropy loss** is not consistent compared to veRL backends
2. Algorithms that require **`compute_advantage_in_trainer=true`** are **not supported**

The complete configuration file can be found at [`tinker.yaml`](tinker.yaml).
