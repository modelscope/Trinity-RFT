# Trinity Benchmark Runner

This Python script is designed to simplify the process of reproducing benchmarks for the **Trinity** system. It provides a command-line interface (CLI) that allows users to easily configure and launch training or inference runs using different datasets, model paths, hardware setups, and algorithm parameters.

---

## 🚀 Features

- Supports both single-node and multi-node distributed training.
- Automates configuration based on user input and cluster resources.
- Works with different datasets (e.g., `gsm8k`, `countdown`).
- Allows custom learning rates, sync intervals, and model configurations.
- Designed to run both locally and in cloud environments like **Aliyun PAI DLC**.

---

## 🛠️ Usage

To run the benchmark, use the following command structure:

```bash
python bench.py <dataset> [options]
```

### Example Command

```bash
python bench.py gsm8k --node_num 1 --gpu_per_node 8 --model_path /path/to/model
```

### Available Arguments

| Argument | Description |
|----------|-------------|
| `dataset` | Dataset name (`gsm8k`, `countdown`) |
| `--dlc` | Use when running in Aliyun PAI DLC environment |
| `--node_num` | Number of nodes in the cluster (default: 1) |
| `--gpu_per_node` | Number of GPUs per node (default: 8) |
| `--vllm_engine_num` | Number of vLLM engines to use |
| `--vllm_tp_size` | Tensor parallel size for vLLM |
| `--explorer_trainer_ratio` | Ratio of explorer engine number to trainer GPU number (default: 0.6), used when `--vllm_engine_num` is not specified |
| `--model_path` | Path to the main model checkpoint |
| `--critic_model_path` | Path to the critic model checkpoint |
| `--taskset_path` | Path to the taskset file |
| `--lr` | Learning rate for actor model |
| `--critic_lr` | Learning rate for critic model |
| `--sync_interval` | Synchronization interval between Trainer and Explorer |

---

## 📂 Output Structure

After running the script, the output will be saved in the `runs/<timestamp>/` directory. The folder contains:

- `config.yaml`: Final configuration used for the run.
- `checkpoints/`: Saved model checkpoints during training.

---

## 📊 Benchmark Results

[WORKING IN PROGRESS]

---

## 🧪 Tips for Reproducing

- Ensure all required models and tasksets are pre-downloaded or accessible at specified paths.
- For multi-node runs, set up proper network communication and shared storage.
- When using vLLM, verify your installation supports it and adjust the tensor parallelism accordingly.
- If using Aliyun PAI DLC, make sure to include the `--dlc` flag.
