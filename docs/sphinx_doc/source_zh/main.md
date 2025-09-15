## 🚀 新闻

* [2025-09] ✨ [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.3.0)] Trinity-RFT v0.3.0 发布：增强的 Buffer、FSDP2 & Megatron 支持，多模态模型，以及全新 RL 算法/示例。
* [2025-08] 🎵 推出 [CHORD](https://github.com/modelscope/Trinity-RFT/tree/main/examples/mix_chord)：动态 SFT + RL 集成，实现进阶 LLM 微调（[论文](https://arxiv.org/pdf/2508.11408)）。
* [2025-08] [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.2.1)] Trinity-RFT v0.2.1 发布。
* [2025-07] [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.2.0)] Trinity-RFT v0.2.0 发布。
* [2025-07] 技术报告（arXiv v2）更新，包含新功能、示例和实验：[链接](https://arxiv.org/abs/2505.17826)。
* [2025-06] [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.1.1)] Trinity-RFT v0.1.1 发布。
* [2025-05] [[发布说明](https://github.com/modelscope/Trinity-RFT/releases/tag/v0.1.0)] Trinity-RFT v0.1.0 发布，同时发布 [技术报告](https://arxiv.org/abs/2505.17826)。
* [2025-04] Trinity-RFT 开源。


## 💡 什么是 Trinity-RFT？

Trinity-RFT 是一个灵活、通用的大语言模型（LLM）强化微调（RFT）框架。它支持广泛的应用场景，并为 [Experience 时代](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) 的 RL 研究提供统一平台。

RFT 流程被模块化为三个核心组件：

* **Explorer**：负责智能体与环境的交互
* **Trainer**：负责模型训练
* **Buffer**：负责数据存储与处理


<img src="https://img.alicdn.com/imgextra/i2/O1CN01H3UbpF1yP7E1OCLbi_!!6000000006570-2-tps-1334-638.png" alt="Trinity-RFT 整体设计" width="800" />



## ✨ 核心特性

* **灵活的 RFT 模式：**
  - 支持同步/异步、on-policy/off-policy 以及在线/离线训练。采样与训练可分离运行，并可在多设备上独立扩展。

  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="Trinity-RFT 支持的 RFT 模式" width="600" />

* **兼容 Agent 框架的工作流：**
  - 支持拼接式和通用多轮智能体工作流。可自动收集来自模型 API 客户端（如 OpenAI）的训练数据，并兼容 AgentScope 等智能体框架。

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="智能体工作流" width="600" />

* **强大的数据流水线：**
  - 支持 rollout 和经验数据的流水线处理，贯穿 RFT 生命周期实现主动管理（优先级、清洗、增强等）。

  <img src="https://img.alicdn.com/imgextra/i2/O1CN01BfeHp61sXSlGjH7zQ_!!6000000005776-2-tps-1734-473.png" alt="数据流水线设计" width="600" />

* **用户友好的框架设计：**
  - 模块化、解耦架构，便于快速上手和二次开发。丰富的图形界面支持低代码使用。

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01Ti0o4320RywoAuyhN_!!6000000006847-2-tps-3840-2134.png" alt="系统架构" width="600" />




## 🛠️ Trinity-RFT 能做什么？

* **用 RL 训练智能体应用**
  - 在 Workflow 中实现智能体-环境交互逻辑  ([示例1](tutorial/example_multi_turn.md)，[示例2](tutorial/example_step_wise.md))，
  - 或直接使用 Agent 框架（如 AgentScope）编写好的工作流 ([示例](tutorial/example_react.md))。

* **快速设计和验证 RL 算法**
  - 在简洁、可插拔的类中开发自定义 RL 算法（损失、采样及其他技巧） ([教程](tutorial/trinity_programming_guide.md#algorithms-for-rl-algorithm-developers)，[示例](tutorial/example_mix_algo.md))。

* **为 RFT 定制数据集和数据流水线**
  - 设计任务定制数据集，构建数据流水线以支持清洗、增强和人类参与场景 ([教程](tutorial/trinity_programming_guide.md#operators-for-data-developers)，[示例](tutorial/example_data_functionalities.md))。


## 致谢


本项目基于许多优秀的开源项目构建，包括：

+ [verl](https://github.com/volcengine/verl) 和 [PyTorch's FSDP](https://pytorch.org/docs/stable/fsdp.html) 用于大模型训练；
+ [vLLM](https://github.com/vllm-project/vllm) 用于大模型推理；
+ [Data-Juicer](https://github.com/modelscope/data-juicer?tab=readme-ov-file) 用于数据处理管道；
+ [AgentScope](https://github.com/modelscope/agentscope) 用于智能体工作流；
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
