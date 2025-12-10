# Automated Context Length Testing for Large Language Models

This script automates the process of determining the **maximum context length** a large language model (LLM) can handle under various distributed training configurations, including different GPU counts and sequence parallelism settings. It iteratively increases the context length during training until an **Out-of-Memory (OOM)** error occurs, logging results and supporting advanced features like RoPE scaling, FSDP strategies, and offloading.

---

## 🧰 Requirements

Ensure Trinity-RFT is well installed ([Installation Guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html)). No extra dependence is required.

---

## 🛠️ Configuration Files

The script relies on two external files:

1. **`context_length.yaml`**
   - Located in the same directory as this script.
   - Defines the base training configuration used by `trinity`.

2. **`workflow/` plugin directory**
   - Contains `CustomWorkflow` expected by the `trinity`, which providing a synthetic training data generator.

Ensure both exist at runtime. You can modify these files to customize the training process.

---

## 🚀 Usage

### Run the Script

```bash
python search_context_length_capacity.py \
  --model_path /path/to/your/model \
  --start_length 4096 \
  --log_dir ./logs \
  --test_gpu_num 1 2 4 \
  --test_sp_num 1 2 \
  --trainer_strategy fsdp \
  --save_hf_checkpoint last \
  --timeout 2400
```

### Required Arguments

| Argument | Description |
|--------|-----------|
| `--model_path` | Path to the pretrained Hugging Face model directory. |

### Optional Arguments

| Argument | Default | Description |
|--------|--------|-----------|
| `--start_length` | `4096` | Initial context length to begin testing. |
| `--log_dir` | `./logs` | Directory to save logs and results. |
| `--checkpoint_path` | `os.environ.get("TRINITY_CHECKPOINT_ROOT_DIR", "./checkpoints/length-test")` | Checkpoint path for testing. Note that this directory will be deleted during the test, please specify a path that is not used by other processes. |
| `--test_gpu_num` | `1 2 4 6` | List of GPU counts to test scalability. |
| `--test_sp_num` | `1` | Sequence parallel group sizes to evaluate. Must divide `test_gpu_num` and number of attention heads. |
| `--save_hf_checkpoint` | `last` | When to save HF format checkpoints (`always`, `never`, `last`). |
| `--entropy_saving` | `False` | Enable memory-saving techniques (if supported). |
| `--offload` | `False` | Offload parameters to CPU to reduce GPU memory usage. |
| `--trainer_strategy` | `fsdp` | Distributed training strategy (`fsdp` or `fsdp2`). |
| `--timeout` | `2400` (40 min) | Maximum time per job before forced termination. |
| `--dlc` | `False` | Specify when running in Aliyun PAI DLC. |

---

## 📂 Output Structure

Logs are saved in a structured hierarchy under `--log_dir`:

```
logs/
└── <model_name>/
    └── gpu-<N>/
        └── sp-<S>/
            └── model_len-<L>.log
```

Each log file corresponds to a specific `(GPU count, SP size, context length)` combination.

Final results are printed to stdout:
```
model_name = Qwen3-0.6B, trainer_gpu_num = 4, sp_num = 2, max_model_len = 40960
```

---

## ⚠️ Notes & Best Practices

- **Model Compatibility**: Ensure the model supports dynamic context extension (e.g., via RoPE scaling).
- **SP Validity**: Only valid SP values (divisors of both GPU count and attention heads) are tested.
- **Checkpoint Root**: Controlled by `TRINITY_CHECKPOINT_ROOT_DIR` env var (default: `./checkpoints/length-test`). Cleared before each trial.
- **Early Termination**: If any run fails due to OOM, the search stops and returns the last successful length.
- **Large Steps After Base Limit**: Basic step size is 4096. And once context exceeds `max_position_embeddings`, step size becomes quarter of original limit.

---

## 🧪 Example: Test Qwen3-0.6B Context Length

```bash
python search_context_length_capacity.py \
  --model_path Qwen/Qwen3-0.6B \
  --test_gpu_num 1 2 4 6 \
  --test_sp_num 1 2 4 \
  --start_length 8192 \
  --log_dir ./results/qwen3-length-scan \
  --trainer_strategy fsdp2 \
  --timeout 3600
```

This command will test the maximum context length for Qwen3-0.6B model with 2, 4, and 8 GPUs, using FSDP2 strategy, and save logs to `./results/qwen3-length-scan`.

---

## 📚 Test Results

Below are empirical results from running this script on various Qwen3 models across different hardware and optimization configurations. These benchmarks help guide configuration choices for maximizing context length within memory constraints.

### Legend
- `*` indicates RoPE scaling (YARN) was applied — context length exceeds the model’s native `max_position_embeddings`.
- `-` indicates OOM occurred even at 4096 context length.
- All tests use `start_length=4096` and increase dynamically.

### A100 80GB

#### Vallina Settings (Baseline)

| #GPU | SP | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
| ---- | -- | ---------- | ---------- | -------- | -------- | --------- |
| 1 | 1 | 20480 | 16384 | - | - | - |
| 2 | 1 | 24576 | 20480 | 12288 | - | - |
| 2 | 2 | 40960 | 40960 | 24576 | - | - |
| 4 | 1 | 24576 | 20480 | 20480 | 8192 | - |
| 4 | 2 | 40960 | 40960 | 36864 | 20480 | - |
| 4 | 4 | 92160* | 81920* | 71680* | 40960 | - |
| 6 | 1 | 24576 | 20480 | 20480 | 12288 | 8192 |
| 6 | 2 | 40960 | 40960 | 40960 | 28672 | 16384 |


#### Enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

> ⚠️ Must be set **before** launching any processes (including Ray clusters).


| #GPU | SP | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
| ---- | -- | ---------- | ---------- | -------- | -------- | --------- |
| 1 | 1 | 24576 | 16384 | - | - | - |
| 2 | 1 | 28672 | 24576 | 16384 | 4096 | - |
| 2 | 2 | 51200* | 40960 | 32768 | - | - |
| 4 | 1 | 28672 | 24576 | 20480 | 12288 | 4096 |
| 4 | 2 | 51200* | 51200* | 40960 | 28672 | 8192 |
| 4 | 4 | 112640* | 102400* | 81920* | 51200* | 20480 |
| 6 | 1 | 28672 | 28672 | 24576 | 16384 | 8192 |
| 6 | 2 | 61440* | 51200* | 40960 | 32768 | 20480 |


#### Enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, FSDP2 Offload and `save_hf_checkpoint=never`

> Uses: `--offload --trainer_strategy fsdp2 --save_hf_checkpoint never`


| #GPU | SP | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
| ---- | -- | ---------- | ---------- | -------- | -------- | --------- |
| 1 | 1 | 28672 | 28672 | 28672 | 24576 | 24576 |
| 2 | 1 | 28672 | 28672 | 28672 | 24576 | 24576 |
| 2 | 2 | 61440* | 51200* | 51200* | 51200* | 40960 |
| 4 | 1 | 28672 | 28672 | 28672 | 24576 | 24576 |
| 4 | 2 | 61440* | 51200* | 51200* | 51200* | 40960 |
| 4 | 4 | 122880* | 112640* | 102400* | 102400* | 92160* |
| 6 | 1 | 28672 | 28672 | 28672 | 24576 | 24576 |
| 6 | 2 | 61440* | 51200* | 51200* | 51200* | 40960 |




### H20 96GB (Higher VRAM, Slower Bandwidth)


#### Vallina Settings


| #GPU | SP | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
| ---- | -- | ---------- | ---------- | -------- | -------- | --------- |
| 1 | 1 | 28672 | 20480 | 8192 | - | - |
| 2 | 1 | 28672 | 24576 | 16384 | 8192 | - |
| 2 | 2 | 51200* | 51200* | 36864 | 16384 | - |
| 4 | 1 | 28672 | 28672 | 24576 | 16384 | 8192 |
| 4 | 2 | 61440* | 51200* | 40960 | 28672 | 16384 |
| 4 | 4 | 112640* | 102400* | 92160* | 51200* | 32768 |
| 6 | 1 | 28672 | 28672 | 24576 | 20480 | 12288 |
| 6 | 2 | 61440* | 51200* | 51200* | 36864 | 24576 |


#### Enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`


| #GPU | SP | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
| ---- | -- | ---------- | ---------- | -------- | -------- | --------- |
| 1 | 1 | 32768 | 24576 | 8192 | - | - |
| 2 | 1 | 36864 | 28672 | 20480 | 8192 | - |
| 2 | 2 | 71680* | 61440* | 40960 | 16384 | - |
| 4 | 1 | 36864 | 32768 | 28672 | 20480 | 8192 |
| 4 | 2 | 71680* | 61440* | 51200* | 36864 | 20480 |
| 4 | 4 | 143360* | 122880* | 102400* | 71680* | 36864 |
| 6 | 1 | 36864 | 32768 | 28672 | 20480 | 16384 |
| 6 | 2 | 71680* | 61440* | 51200* | 40960 | 32768 |



#### Enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and FSDP2 Offload

> Uses: `--offload --trainer_strategy fsdp2`

| #GPU | SP | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
| ---- | -- | ---------- | ---------- | -------- | -------- | --------- |
| 1 | 1 | 36864 | 36864 | 32768 | 28672 | 28672 |
| 2 | 1 | 36864 | 36864 | 32768 | 28672 | 28672 |
| 2 | 2 | 71680* | 61440* | 61440* | 61440 | 51200* |
| 4 | 1 | 36864 |  | 32768 | 28672 | 28672 |
| 4 | 2 | 71680* | 71680* | 61440* | 61440* | |
| 4 | 4 | 143360* | 133120* | 133120* | 122880* | 112640* |
| 6 | 1 | 36864 |  | 32768 | 28672 | 28672 |
| 6 | 2 | 71680* | 71680* | 61440* | 61440* | 51200* |


#### Enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, FSDP2 Offload and `save_hf_checkpoint=never`

> Uses: `--offload --trainer_strategy fsdp2 --save_hf_checkpoint never`


| #GPU | SP | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
| ---- | -- | ---------- | ---------- | -------- | -------- | --------- |
| 1 | 1 | 36864 | 36864 | 32768 | 28672 | 28672 |
| 2 | 1 | 36864 | 36864 | 32768 | 28672 | 28672 |
| 2 | 2 | 71680* | 61440* | 61440* | 61440* | |
| 4 | 1 | 36864 |  | 32768 | 28672 | 28672 |
| 4 | 2 | 71680* | 71680* | 61440* | 61440* | 51200* |
| 4 | 4 | 143360* | 133120* | 133120* | 122880* | 112640* |
| 6 | 1 | 36864 |  | 32768 | 28672 | 28672 |
| 6 | 2 | 71680* | 71680* | 61440* | 61440* | 51200* |
