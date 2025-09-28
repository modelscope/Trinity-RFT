[**English Homepage**](https://github.com/modelscope/Trinity-RFT/blob/main/README.md) | [**中文文档**](https://modelscope.github.io/Trinity-RFT/zh/) | [**常见问题**](./docs/sphinx_doc/source/zh/tutorial/faq.md)

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

## 💡 什么是 Trinity-RFT ?

Trinity-RFT 是一个灵活、通用的大语言模型（LLM）强化微调（RFT）框架。 其将 RFT 流程解耦为三个关键模块：**Explorer**、**Trainer** 和 **Buffer**，并面向不同背景和目标的用户提供相应功能：

* 🤖 面向智能体应用开发者。[[教程]](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/develop_workflow.html)
  - 训练智能体应用，以增强其在指定环境中完成任务的能力
  - 示例：[多轮交互](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_multi_turn.html)，[ReAct 智能体](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_react.html)

* 🧠 面向 RL 算法研究者。[[教程]](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/develop_algorithm.html)
  - 在简洁、可插拔的类中设计和验证新的 RL 算法
  - 示例：[SFT/GRPO混合算法](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_mix_algo.html)

* 📊 面向数据工程师。[[教程]](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/develop_operator.html)
  - 设计任务定制数据集，构建数据流水线以支持清洗、增强和人类参与场景
  - 示例：[数据处理](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_data_functionalities.html)

# 🌟 核心特性

* **灵活的 RFT 模式：**
  - 支持同步/异步、on-policy/off-policy 以及在线/离线训练。采样与训练可分离运行，并可在多设备上独立扩展。

  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="Trinity-RFT 支持的 RFT 模式" width="600" />

* **通用 Agentic-RL：**
  - 支持拼接式和通用多轮交互，能够直接训练使用 AgentScope 等智能体框架开发的 Agent 应用。

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="智能体工作流" width="600" />

* **全流程的数据流水线：**
  - 支持 rollout 和经验数据的流水线处理，贯穿 RFT 生命周期实现主动管理（优先级、清洗、增强等）。

  <img src="https://img.alicdn.com/imgextra/i2/O1CN01BfeHp61sXSlGjH7zQ_!!6000000005776-2-tps-1734-473.png" alt="数据流水线设计" width="600" />

* **用户友好的框架设计：**
  - 模块化、解耦架构，便于快速上手和二次开发。丰富的图形界面支持低代码使用。

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01Ti0o4320RywoAuyhN_!!6000000006847-2-tps-3840-2134.png" alt="系统架构" width="600" />


## 🚀 新闻

* [2025-09] ✨ [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.3.0)] Trinity-RFT v0.3.0 发布：增强的 Buffer、FSDP2 & Megatron 支持，多模态模型，以及全新 RL 算法/示例。
* [2025-08] 🎵 推出 [CHORD](https://github.com/modelscope/Trinity-RFT/tree/main/examples/mix_chord)：动态 SFT + RL 集成，实现进阶 LLM 微调（[论文](https://arxiv.org/pdf/2508.11408)）。
* [2025-08] [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.2.1)] Trinity-RFT v0.2.1 发布。
* [2025-07] [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.2.0)] Trinity-RFT v0.2.0 发布。
* [2025-07] 技术报告（arXiv v2）更新，包含新功能、示例和实验：[链接](https://arxiv.org/abs/2505.17826)。
* [2025-06] [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.1.1)] Trinity-RFT v0.1.1 发布。
* [2025-05] [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.1.0)] Trinity-RFT v0.1.0 发布，同时发布 [技术报告](https://arxiv.org/abs/2505.17826)。
* [2025-04] Trinity-RFT 开源。

---

## 目录


- [快速上手](#快速上手)
  - [第一步：安装](#第一步安装)
  - [第二步：准备数据集和模型](#第二步准备数据集和模型)
  - [第三步：准备配置文件](#第三步准备配置文件)
  - [第四步：运行 RFT 流程](#第四步运行-rft-流程)
- [更多教程](#更多教程)
- [开发路线图](#开发路线图)
- [贡献指南](#贡献指南)
- [致谢](#致谢)
- [引用](#引用)



## 快速上手


> [!NOTE]
> 本项目正处于活跃开发阶段。欢迎提出意见和建议！


### 第一步：安装

在安装之前，请确保您的系统满足以下要求：

- **Python**：版本 3.10 至 3.12（含）
- **CUDA**：版本 12.4 至 12.8（含）
- **GPU**：至少 2 块 GPU

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

## 通过 PyPI 安装

如果您只需使用 Trinity-RFT 而不打算修改代码：

```bash
pip install trinity-rft==0.3.0
pip install flash-attn==2.8.1
```

或使用 `uv`：

```bash
uv pip install trinity-rft==0.3.0
uv pip install flash-attn==2.8.1
```

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

> 如需使用 **Megatron-LM** 进行训练，请参考 [Megatron-LM 支持](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_megatron.html)


### 第二步：准备数据集和模型


Trinity-RFT 支持来自 Huggingface 和 ModelScope 的大多数数据集和模型。


**准备模型**，保存到本地目录 `$MODEL_PATH/{model_name}`：

```bash
# 使用 Huggingface
huggingface-cli download {model_name} --local-dir $MODEL_PATH/{model_name}

# 使用 ModelScope
modelscope download {model_name} --local_dir $MODEL_PATH/{model_name}
```

更多关于模型下载的细节，请参考 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 或  [ModelScope](https://modelscope.cn/docs/models/download)。



**准备数据集**，保存到本地目录 `$DATASET_PATH/{dataset_name}`：

```bash
# 使用 Huggingface
huggingface-cli download {dataset_name} --repo-type dataset --local-dir $DATASET_PATH/{dataset_name}

# 使用 ModelScope
modelscope download --dataset {dataset_name} --local_dir $DATASET_PATH/{dataset_name}
```

更多关于数据集下载的细节，请参考 [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space) 或 [ModelScope](https://modelscope.cn/docs/datasets/download)。



### 第三步：准备配置文件


Trinity-RFT 提供了一个 Web 界面来配置您的 RFT 流程。

> [!NOTE]
> 这是一个实验性功能，我们将持续改进。


要启动 Web 界面进行配置，您可以运行：

```bash
trinity studio --port 8080
```

然后您可以在网页上配置您的 RFT 流程并生成一个配置文件。您可以保存该配置文件以备后用，或按照下一节的描述直接运行。

高阶用户也可以直接编辑配置文件。
我们在 [`examples`](examples/) 目录中提供了一些示例配置文件。

若需完整的 GUI 功能，请参考 [Trinity-Studio](https://github.com/modelscope/Trinity-Studio) 仓库。


<details>

<summary> 示例：配置管理器 GUI </summary>

![config-manager](https://img.alicdn.com/imgextra/i1/O1CN01yhYrV01lGKchtywSH_!!6000000004791-2-tps-1480-844.png)


</details>




### 第四步：运行 RFT 流程


启动一个 Ray 集群：

```shell
# 在主节点上
ray start --head

# 在工作节点上
ray start --address=<master_address>
```

（可选）登录 [wandb](https://docs.wandb.ai/quickstart/) 以便更好地监控 RFT 过程：

```shell
export WANDB_API_KEY=<your_api_key>
wandb login
```

对于命令行用户，运行 RFT 流程：

```shell
trinity run --config <config_path>
```

例如，以下是在 GSM8k 数据集上使用 GRPO 微调 Qwen2.5-1.5B-Instruct 的命令：

```shell
trinity run --config examples/grpo_gsm8k/gsm8k.yaml
```

对于 Studio 用户，在 Web 界面中点击“运行”。


## 更多教程

> [!NOTE]
> 更多教程请参考 [Trinity-RFT 文档](https://modelscope.github.io/Trinity-RFT/)。

运行不同 RFT 模式的教程：

+ [快速开始：在 GSM8k 上运行 GRPO](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_basic.html)
+ [Off-Policy RFT](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_advanced.html)
+ [全异步 RFT](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_async_mode.html)
+ [通过 DPO 或 SFT 进行离线学习](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_dpo.html)


将 Trinity-RFT 适配到新的多轮智能体场景的教程：

+ [拼接多轮任务](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_multi_turn.html)
+ [通用多轮任务](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_step_wise.html)
+ [调用智能体框架中的 ReAct 工作流](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_react.html)


数据相关功能的教程：

+ [高级数据处理及 Human-in-the-loop](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_data_functionalities.html)


使用 Trinity-RFT 进行 RL 算法开发/研究的教程：

+ [使用 Trinity-RFT 进行 RL 算法开发](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/example_mix_algo.html)


完整配置指南：

+ 请参阅[此文档](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/trinity_configs.html)


面向开发者和研究人员的指南：

+ [用于快速验证实验的 Benchmark 工具](./benchmark/README.md)
+ [理解 explorer-trainer 同步逻辑](https://modelscope.github.io/Trinity-RFT/zh/main/tutorial/synchronizer.html)



## 开发路线图

路线图：[#51](https://github.com/modelscope/Trinity-RFT/issues/51)



## 贡献指南


本项目正处于活跃开发阶段，我们欢迎来自社区的贡献！


请参阅 [贡献指南](./CONTRIBUTING.md) 了解详情。


## 致谢


本项目基于许多优秀的开源项目构建，包括：

+ [verl](https://github.com/volcengine/verl) 和 [PyTorch's FSDP](https://pytorch.org/docs/stable/fsdp.html) 用于大模型训练；
+ [vLLM](https://github.com/vllm-project/vllm) 用于大模型推理；
+ [Data-Juicer](https://github.com/modelscope/data-juicer?tab=readme-ov-file) 用于数据处理管道；
+ [AgentScope](https://github.com/agentscope-ai/agentscope) 用于智能体工作流；
+ [Ray](https://github.com/ray-project/ray) 用于分布式系统；
+ 我们也从 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[TRL](https://github.com/huggingface/trl) 和 [ChatLearn](https://github.com/alibaba/ChatLearn) 等框架中汲取了灵感；
+ ......

## 引用


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
