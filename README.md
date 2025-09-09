[**中文主页**](https://github.com/modelscope/Trinity-RFT/blob/main/README_zh.md) | [**Tutorial**](https://modelscope.github.io/Trinity-RFT/) | [**FAQ**](./docs/sphinx_doc/source/tutorial/faq.md)

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01lvLpfw25Pl4ohGZnU_!!6000000007519-2-tps-1628-490.png" alt="Trinity-RFT" style="height: 120px;">
</div>



<h2 align="center">Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models</h2>


<div align="center">

[![paper](http://img.shields.io/badge/cs.LG-2505.17826-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.17826)
[![doc](https://img.shields.io/badge/Docs-blue?logo=markdown)](https://modelscope.github.io/Trinity-RFT/)
[![pypi](https://img.shields.io/pypi/v/trinity-rft?logo=pypi&color=026cad)](https://pypi.org/project/trinity-rft/)
![license](https://img.shields.io/badge/license-Apache--2.0-000000.svg)

</div>


## 🚀 News

* [2025-09] ✨ [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.3.0)] Trinity-RFT v0.3.0 released: enhanced Buffer, FSDP2 & Megatron support, multi-modal models, and new RL algorithms/examples.
* [2025-08] 🎵 Introducing [CHORD](https://github.com/modelscope/Trinity-RFT/tree/main/examples/mix_chord): dynamic SFT + RL integration for advanced LLM fine-tuning ([paper](https://arxiv.org/pdf/2508.11408)).
* [2025-08] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.2.1)] Trinity-RFT v0.2.1 released.
* [2025-07] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.2.0)] Trinity-RFT v0.2.0 released.
* [2025-07] Technical report (arXiv v2) updated with new features, examples, and experiments: [link](https://arxiv.org/abs/2505.17826).
* [2025-06] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.1.1)] Trinity-RFT v0.1.1 released.
* [2025-05] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.1.0)] Trinity-RFT v0.1.0 released, plus [technical report](https://arxiv.org/abs/2505.17826).
* [2025-04] Trinity-RFT open sourced.


## 💡 What is Trinity-RFT?

Trinity-RFT is a flexible, general-purpose framework for reinforcement fine-tuning (RFT) of large language models (LLMs). It supports a wide range of applications and provides a unified platform for RL research in the [era of experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf).

The RFT process is modularized into three core components:

* **Explorer**: Handles agent-environment interaction
* **Trainer**: Manages model training
* **Buffer**: Manages data storage and processing


<img src="https://img.alicdn.com/imgextra/i2/O1CN01H3UbpF1yP7E1OCLbi_!!6000000006570-2-tps-1334-638.png" alt="The high-level design of Trinity-RFT" width="800" />



## ✨ Key Features

* **Flexible RFT Modes:**
  - Supports synchronous/asynchronous, on-policy/off-policy, and online/offline training. Rollout and training can run separately and scale independently across devices.

  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="RFT modes supported by Trinity-RFT" width="600" />

* **Agent Framework Compatible Workflows:**
  - Supports both concatenated and general multi-turn agentic workflows. Automatically collects training data from model API clients (e.g., OpenAI) and is compatible with agent frameworks like AgentScope.

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="Agentic workflows" width="600" />

* **Powerful Data Pipelines:**
  - Enables pipeline processing of rollout and experience data, supporting active management (prioritization, cleaning, augmentation) throughout the RFT lifecycle.

  <img src="https://img.alicdn.com/imgextra/i2/O1CN01BfeHp61sXSlGjH7zQ_!!6000000005776-2-tps-1734-473.png" alt="Data pipeline design" width="600" />

* **User-Friendly Design:**
  - Modular, decoupled architecture for easy adoption and development. Rich graphical user interfaces enable low-code usage.

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01Ti0o4320RywoAuyhN_!!6000000006847-2-tps-3840-2134.png" alt="System architecture" width="600" />




## 🛠️ What can I use Trinity-RFT for?

* **Train agent applications with RL and minimal migration cost** [[Tutorial]](https://modelscope.github.io/Trinity-RFT/main/tutorial/trinity_programming_guide.html#workflows-for-rl-environment-developers)
  - Implement agent-environment interaction logic in a single workflow class ([example1](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_multi_turn.html), [example2](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_step_wise.html)),
  - Or import workflows from agent frameworks like AgentScope ([example](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_react.html)).

* **Rapid RL algorithm design and validation** [[Tutorial]](https://modelscope.github.io/Trinity-RFT/main/tutorial/trinity_programming_guide.html#algorithms-for-rl-algorithm-developers)
  - Develop custom RL algorithms (loss design, sampling strategy, etc.) in compact, plug-and-play classes ([example](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_mix_algo.html)).

* **Custom datasets and data pipelines for RFT** [[Tutorial]](https://modelscope.github.io/Trinity-RFT/main/tutorial/trinity_programming_guide.html#operators-for-data-developers)
  - Design task-specific datasets and build data pipelines for cleaning, augmentation, and human-in-the-loop scenarios ([example](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_data_functionalities.html)).

---

## Table of contents


- [Getting started](#getting-started)
  - [Step 1: installation](#step-1-installation)
  - [Step 2: prepare dataset and model](#step-2-prepare-dataset-and-model)
  - [Step 3: configurations](#step-3-configurations)
  - [Step 4: run the RFT process](#step-4-run-the-rft-process)
- [Further tutorials](#further-tutorials)
- [Upcoming features](#upcoming-features)
- [Contribution guide](#contribution-guide)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)



## Getting started


> [!NOTE]
> This project is currently under active development. Comments and suggestions are welcome!


### Step 1: installation

#### Prerequisites

Before installing, make sure your system meets the following requirements:

- **Python**: version 3.10 to 3.12 (inclusive)
- **CUDA**: version 12.4 to 12.8 (inclusive)
- **GPUs**: at least 2 GPUs


#### Option A: Install from Source (Recommended)

This method gives you full control and is best if you plan to customize or contribute to the project.

##### 1. Clone the Repository

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT
```

##### 2. Set Up a Virtual Environment

Choose one of the following options to create an isolated environment:

###### Using Conda
```bash
conda create -n trinity python=3.10
conda activate trinity
```

###### Using venv
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

##### 3. Install the Package

Install in editable mode so you can make changes without reinstalling:

```bash
pip install -e ".[dev]"
```

##### 4. Install Flash Attention

Flash Attention boosts training speed. It takes a few minutes to compile — please be patient!

```bash
pip install flash-attn==2.8.1
```

If you encounter issues during installation, try this alternative:

```bash
pip install flash-attn==2.8.1 --no-build-isolation
```


##### ⚡ Fast Alternative: Use `uv` (Optional)

If you'd like a faster installation, try [`uv`](https://github.com/astral-sh/uv), a modern Python package installer:

```bash
uv venv
source .venv/bin/activate

uv pip install -e ".[dev]"
uv pip install flash-attn==2.8.1 --no-build-isolation
```

#### Option B: Install via pip (Quick Start)

If you just want to use the package without modifying the code:

```bash
pip install trinity-rft==0.3.0
pip install flash-attn==2.8.1  # Install Flash Attention separately

# Use uv to install trinity-rft
# uv pip install trinity-rft==0.3.0
# uv pip install flash-attn==2.8.1
```

#### Option C: Use Docker

We provide a Docker setup for hassle-free environment configuration.

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

## Build the Docker image
## Tip: You can modify the Dockerfile to add mirrors or set API keys
docker build -f scripts/docker/Dockerfile -t trinity-rft:latest .

## Run the container
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v <path_to_your_data_and_checkpoints>:/data \
  trinity-rft:latest
```

💡 **Note**: Replace `<path_to_your_data_and_checkpoints>` with the actual path on your machine where datasets and model checkpoints are stored.

> If you'd like to integrate with **Megatron-LM**, check out our [example setup guide for Megatron](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_megatron.html).

### Step 2: prepare dataset and model


Trinity-RFT supports most datasets and models from Huggingface and ModelScope.


**Prepare the model** in the local directory `$MODEL_PATH/{model_name}`:

```bash
# Using Huggingface
huggingface-cli download {model_name} --local-dir $MODEL_PATH/{model_name}

# Using Modelscope
modelscope download {model_name} --local_dir $MODEL_PATH/{model_name}
```

For more details about model downloading, see [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) or [ModelScope](https://modelscope.cn/docs/models/download).



**Prepare the dataset** in the local directory `$DATASET_PATH/{dataset_name}`:

```bash
# Using Huggingface
huggingface-cli download {dataset_name} --repo-type dataset --local-dir $DATASET_PATH/{dataset_name}

# Using Modelscope
modelscope download --dataset {dataset_name} --local_dir $DATASET_PATH/{dataset_name}
```

For more details about dataset downloading, see [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space) or [ModelScope](https://modelscope.cn/docs/datasets/download).



### Step 3: configurations


Trinity-RFT provides a web interface for configuring your RFT process.

> [!NOTE]
> This is an experimental feature, and we will continue to improve it.


To launch the web interface for minimal configurations, you can run

```bash
trinity studio --port 8080
```

Then you can configure your RFT process in the web page and generate a config file. You can save the config file for later use or run it directly as described in the following section.

Advanced users can also edit the config file directly.
We provide example config files in [`examples`](examples/).

For complete GUI features, please refer to the monorepo for [Trinity-Studio](https://github.com/modelscope/Trinity-Studio).


<details>

<summary> Example: config manager GUI </summary>

![config-manager](https://img.alicdn.com/imgextra/i1/O1CN01yhYrV01lGKchtywSH_!!6000000004791-2-tps-1480-844.png)


</details>




### Step 4: run the RFT process


Start a ray cluster:

```shell
# On master node
ray start --head

# On worker nodes
ray start --address=<master_address>
```

(Optional) Log in to [wandb](https://docs.wandb.ai/quickstart/) for better monitoring:

```shell
export WANDB_API_KEY=<your_api_key>
wandb login
```

For command-line users, run the RFT process:

```shell
trinity run --config <config_path>
```

For example, below is the command for fine-tuning Qwen2.5-1.5B-Instruct on GSM8k with GRPO:

```shell
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```

For studio users, click "Run" in the web interface.


## Further tutorials

> [!NOTE]
> For more tutorials, please refer to the [Trinity-RFT Documentation](https://modelscope.github.io/Trinity-RFT/).


Tutorials for running different RFT modes:

+ [Quick example: GRPO on GSM8k](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_reasoning_basic.html)
+ [Off-policy RFT](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_reasoning_advanced.html)
+ [Fully asynchronous RFT](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_async_mode.html)
+ [Offline learning by DPO or SFT](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_dpo.html)


Tutorials for adapting Trinity-RFT to multi-step agentic scenarios:

+ [Concatenated multi-turn workflow](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_multi_turn.html)
+ [General multi-step workflow](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_step_wise.html)
+ [ReAct workflow with an agent framework](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_react.html)


Tutorials for data-related functionalities:

+ [Advanced data processing & human-in-the-loop](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_data_functionalities.html)


Tutorials for RL algorithm development/research with Trinity-RFT:

+ [RL algorithm development with Trinity-RFT](https://modelscope.github.io/Trinity-RFT/main/tutorial/example_mix_algo.html)


Guidelines for full configurations:

+ See [this document](https://modelscope.github.io/Trinity-RFT/main/tutorial/trinity_configs.html)


Guidelines for developers and researchers:

+ [Benchmark Toolkit for quick verification and experimentation](./benchmark/README.md)
+ [Understand the coordination between explorer and trainer](https://modelscope.github.io/Trinity-RFT/main/tutorial/synchronizer.html)


## Upcoming features

A tentative roadmap: [#51](https://github.com/modelscope/Trinity-RFT/issues/51)


## Contribution guide

This project is currently under active development, and we welcome contributions from the community!

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed contribution guidelines.


## Acknowledgements

This project is built upon many excellent open-source projects, including:

+ [verl](https://github.com/volcengine/verl) and [PyTorch's FSDP](https://pytorch.org/docs/stable/fsdp.html) for LLM training;
+ [vLLM](https://github.com/vllm-project/vllm) for LLM inference;
+ [Data-Juicer](https://github.com/modelscope/data-juicer?tab=readme-ov-file) for data processing pipelines;
+ [AgentScope](https://github.com/modelscope/agentscope) for agentic workflow;
+ [Ray](https://github.com/ray-project/ray) for distributed systems;
+ we have also drawn inspirations from RL frameworks like [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [TRL](https://github.com/huggingface/trl) and [ChatLearn](https://github.com/alibaba/ChatLearn);
+ ......


## Citation

```bibtex
@misc{trinity-rft,
      title={Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models},
      author={Xuchen Pan and Yanxi Chen and Yushuo Chen and Yuchang Sun and Daoyuan Chen and Wenhao Zhang and Yuexiang Xie and Yilun Huang and Yilei Zhang and Dawei Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2505.17826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17826},
}
```
