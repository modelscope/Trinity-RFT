[**中文主页**](https://github.com/modelscope/Trinity-RFT/blob/main/README_zh.md) | [**Tutorial**](https://modelscope.github.io/Trinity-RFT/) | [**FAQ**](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/faq.html)

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

## 💡 What is Trinity-RFT?

Trinity-RFT is a flexible, general-purpose framework for reinforcement fine-tuning (RFT) of large language models (LLMs). It decouples the RFT process into three key components: **Explorer**, **Trainer**, and **Buffer**, and provides functionalities for users with different backgrounds and objectives:


* 🤖 For agent application developers. [[tutorial]](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/develop_workflow.html)
  - Train agent applications to improve their ability to complete tasks in specific environments.
  - Examples: [Multi-Turn Interaction](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_multi_turn.html), [ReAct Agent](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_react.html)

* 🧠 For RL algorithm researchers. [[tutorial]](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/develop_algorithm.html)
  - Design and validate new reinforcement learning algorithms using compact, plug-and-play modules.
  - Example: [Mixture of SFT and GRPO](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_mix_algo.html)

* 📊 For data engineers. [[tutorial]](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/develop_operator.html)
  - Create datasets and build data pipelines for cleaning, augmentation, and human-in-the-loop scenarios.
  - Example: [Data Processing](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_data_functionalities.html)


## 🌟 Key Features

* **Flexible RFT Modes:**
  - Supports synchronous/asynchronous, on-policy/off-policy, and online/offline RL.
  - Rollout and training can run separately and scale independently across devices.
  - Boost sample and time efficiency by experience replay.

  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="RFT modes supported by Trinity-RFT" width="600" />

* **Agentic RL Support:**
  - Supports both concatenated and general multi-step agentic workflows.
  - Able to directly train agent applications developed using agent frameworks like AgentScope.

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="Agentic workflows" width="600" />

* **Full-Lifecycle Data Pipelines:**
  - Enables pipeline processing of rollout tasks and experience samples.
  - Active data management (e.g., prioritization, cleaning, augmentation) throughout the RFT lifecycle.
  - Native support for multi-task joint learning.

  <img src="https://img.alicdn.com/imgextra/i2/O1CN01Gk9CRw28NsL09nbOj_!!6000000007921-2-tps-2530-660.png" alt="Data pipeline design" width="720" />

* **User-Friendly Design:**
  - Plug-and-play modules and decoupled architecture, facilitating easy adoption and development.
  - Rich graphical user interfaces enable low-code usage.

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01Ti0o4320RywoAuyhN_!!6000000006847-2-tps-3840-2134.png" alt="System architecture" width="600" />


## 🔨 Tutorials and Guidelines


| Category | Tutorial / Guideline |
| --- | --- |
| Run diverse RFT modes | + [Quick example: GRPO on GSM8k](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_basic.html)<br>+ [Off-policy RFT](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_reasoning_advanced.html)<br>+ [Fully asynchronous RFT](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_async_mode.html)<br>+ [Offline learning by DPO or SFT](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_dpo.html) |
| Multi-step agentic scenarios | + [Concatenated multi-turn workflow](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_multi_turn.html)<br>+ [General multi-step workflow](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_step_wise.html)<br>+ [ReAct workflow with an agent framework](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_react.html) |
| Advanced data pipelines | + [Rollout task mixing and selection](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/develop_selector.html)<br>+ [Experience replay](https://github.com/modelscope/Trinity-RFT/tree/main/examples/ppo_countdown_exp_replay)<br>+ [Advanced data processing & human-in-the-loop](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_data_functionalities.html) |
| Algorithm development / research | + [RL algorithm development with Trinity-RFT](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_mix_algo.html) ([paper](https://arxiv.org/pdf/2508.11408))<br>+ Non-verifiable domains: [RULER](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k_ruler), [trainable RULER](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k_trainable_ruler), [rubric-as-reward](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_rubric_as_reward) <br>+ [Research project: group-relative REINFORCE](https://github.com/modelscope/Trinity-RFT/tree/main/examples/rec_gsm8k) ([paper](https://arxiv.org/abs/2509.24203))|
| Going deeper into Trinity-RFT | + [Full configurations](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html)<br>+ [Benchmark toolkit for quick verification and experimentation](./benchmark/README.md)<br>+ [Understand the coordination between explorer and trainer](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/synchronizer.html) |


> [!NOTE]
> For more tutorials, please refer to the [Trinity-RFT documentation](https://modelscope.github.io/Trinity-RFT/).


## 🚀 News

* [2025-10] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.3.2)] Trinity-RFT v0.3.2 released: bug fixes and advanced task selection & scheduling.
* [2025-10] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.3.1)] Trinity-RFT v0.3.1 released: multi-stage training support, improved agentic RL examples, LoRA support, debug mode and new RL algorithms.
* [2025-09] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.3.0)] Trinity-RFT v0.3.0 released: enhanced Buffer, FSDP2 & Megatron support, multi-modal models, and new RL algorithms/examples.
* [2025-08] Introducing [CHORD](https://github.com/modelscope/Trinity-RFT/tree/main/examples/mix_chord): dynamic SFT + RL integration for advanced LLM fine-tuning ([paper](https://arxiv.org/pdf/2508.11408)).
* [2025-08] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.2.1)] Trinity-RFT v0.2.1 released.
* [2025-07] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.2.0)] Trinity-RFT v0.2.0 released.
* [2025-07] Technical report (arXiv v2) updated with new features, examples, and experiments: [link](https://arxiv.org/abs/2505.17826).
* [2025-06] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.1.1)] Trinity-RFT v0.1.1 released.
* [2025-05] [[Release Notes](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.1.0)] Trinity-RFT v0.1.0 released, plus [technical report](https://arxiv.org/abs/2505.17826).
* [2025-04] Trinity-RFT open sourced.


---

## Table of Contents

- [Quick Start](#quick-start)
  - [Step 1: installation](#step-1-installation)
  - [Step 2: prepare dataset and model](#step-2-prepare-dataset-and-model)
  - [Step 3: configurations](#step-3-configurations)
  - [Step 4: run the RFT process](#step-4-run-the-rft-process)
- [Contribution guide](#contribution-guide)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)



## Quick Start


> [!NOTE]
> This project is currently under active development. Comments and suggestions are welcome!


### Step 1: installation

Before installing, make sure your system meets the following requirements:

- **Python**: version 3.10 to 3.12 (inclusive)
- **CUDA**: version >= 12.6
- **GPUs**: at least 2 GPUs


#### From Source (Recommended)

If you plan to customize or contribute to Trinity-RFT, this is the best option.

##### 1. Clone the Repository

```bash
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT
```

##### 2. Set Up a Virtual Environment

Choose one of the following options:

###### Using Conda

```bash
conda create -n trinity python=3.10
conda activate trinity

pip install -e ".[dev]"
pip install -e ".[flash_attn]"
# if you encounter issues when installing flash-attn, try:
# pip install flash-attn==2.8.1 --no-build-isolation
```

###### Using venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pip install -e ".[flash_attn]"
# if you encounter issues when installing flash-attn, try:
# pip install flash-attn==2.8.1 --no-build-isolation
```

###### Using `uv`

[`uv`](https://github.com/astral-sh/uv) is a modern Python package installer.

```bash
uv sync --extra dev --extra flash_attn
```


#### Via PyPI

If you just want to use the package without modifying the code:

```bash
pip install trinity-rft
pip install flash-attn==2.8.1
```

Or with `uv`:

```bash
uv pip install trinity-rft
uv pip install flash-attn==2.8.1
```


#### Using Docker

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

> For training with **Megatron-LM**, please refer to [Megatron-LM Backend](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/example_megatron.html).

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

(Optional) You may use [Wandb](https://docs.wandb.ai/quickstart/) / [TensorBoard](https://www.tensorflow.org/tensorboard) / [MLFlow](https://mlflow.org) for better monitoring.
Please refer to [this documentation](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html#monitor-configuration) for the corresponding configurations.
For example, to log in to Wandb:

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



## Contribution Guide

This project is currently under active development, and we welcome contributions from the community!

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed contribution guidelines.


## Acknowledgements

This project is built upon many excellent open-source projects, including:

+ [verl](https://github.com/volcengine/verl), [FSDP](https://pytorch.org/docs/stable/fsdp.html) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for LLM training;
+ [vLLM](https://github.com/vllm-project/vllm) for LLM inference;
+ [Data-Juicer](https://github.com/modelscope/data-juicer?tab=readme-ov-file) for data processing pipelines;
+ [AgentScope](https://github.com/agentscope-ai/agentscope) for agentic workflow;
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
