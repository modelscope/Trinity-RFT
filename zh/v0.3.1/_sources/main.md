## 💡 什么是 Trinity-RFT？

Trinity-RFT 是一个灵活、通用的大语言模型（LLM）强化微调（RFT）框架。 其将 RFT 流程解耦为三个关键模块：**Explorer**、**Trainer** 和 **Buffer**，并面向不同背景和目标的用户提供相应功能：

* 🤖 面向智能体应用开发者。[[教程]](/tutorial/develop_workflow.md)
  - 训练智能体应用，以增强其在指定环境中完成任务的能力
  - 示例：[多轮交互](/tutorial/example_multi_turn.md)，[ReAct 智能体](/tutorial/example_react.md)

* 🧠 面向 RL 算法研究者。[[教程]](/tutorial/develop_algorithm.md)
  - 在简洁、可插拔的类中设计和验证新的 RL 算法
  - 示例：[SFT/GRPO混合算法](/tutorial/example_mix_algo.md)

* 📊 面向数据工程师。[[教程]](/tutorial/develop_operator.md)
  - 设计任务定制数据集，构建数据流水线以支持清洗、增强和人类参与场景
  - 示例：[数据处理](/tutorial/example_data_functionalities.md)

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
