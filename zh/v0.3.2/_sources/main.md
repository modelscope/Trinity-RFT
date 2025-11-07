## 💡 什么是 Trinity-RFT？

Trinity-RFT 是一个灵活、通用的大语言模型（LLM）强化微调（RFT）框架。 其将 RFT 流程解耦为三个关键模块：**Explorer**、**Trainer** 和 **Buffer**，并面向不同背景和目标的用户提供相应功能：

* 🤖 面向智能体应用开发者。[[教程]](/tutorial/develop_workflow.md)
  - 训练智能体应用，以增强其在指定环境中完成任务的能力
  - 示例：[多轮交互](/tutorial/example_multi_turn.md)，[ReAct 智能体](/tutorial/example_react.md)

* 🧠 面向 RL 算法研究者。[[教程]](/tutorial/develop_algorithm.md)
  - 在简洁、可插拔的类中设计和验证新的 RL 算法
  - 示例：[SFT/RL 混合算法](/tutorial/example_mix_algo.md)

* 📊 面向数据工程师。[[教程]](/tutorial/develop_operator.md)
  - 设计针对任务定制的数据集，构建处理流水线以支持数据清洗、增强以及人类参与场景
  - 示例：[数据处理基础](/tutorial/example_data_functionalities.md)，[在线任务选择](https://github.com/modelscope/Trinity-RFT/tree/main/examples/bots)

# 🌟 核心特性

* **灵活的 RFT 模式：**
  - 支持同步/异步、on-policy/off-policy 以及在线/离线强化学习
  - 采样与训练可分离运行，并可在多设备上独立扩展
  - 支持经验回放，进一步提升样本与时间效率

  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="Trinity-RFT 支持的 RFT 模式" width="600" />

* **Agentic RL 支持：**
  - 支持拼接式多轮和通用多轮交互
  - 能够直接训练使用 AgentScope 等智能体框架开发的 Agent 应用

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="智能体工作流" width="600" />

* **全流程的数据流水线：**
  - 支持 rollout 任务和经验数据的流水线处理
  - 贯穿 RFT 生命周期的主动数据管理（优先级排序、清洗、增强等）
  - 原生支持多任务联合训练

  <img src="https://img.alicdn.com/imgextra/i2/O1CN01Gk9CRw28NsL09nbOj_!!6000000007921-2-tps-2530-660.png" alt="数据流水线设计" width="600" />

* **用户友好的框架设计：**
  - 即插即用模块与解耦式架构，便于快速上手和二次开发
  - 丰富的图形界面，支持低代码使用

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01Ti0o4320RywoAuyhN_!!6000000006847-2-tps-3840-2134.png" alt="系统架构" width="600" />




## 🔨 教程与指南


| Category | Tutorial / Guideline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 运行各种 RFT 模式 | + [快速开始：在 GSM8k 上运行 GRPO](/tutorial/example_reasoning_basic.md)<br>+ [Off-policy RFT](/tutorial/example_reasoning_advanced.md)<br>+ [全异步 RFT](/tutorial/example_async_mode.md)<br>+ [通过 DPO 或 SFT 进行离线学习](/tutorial/example_dpo.md)                                                                                                                                                                                                                                                                                                                                             |
| 多轮智能体场景 | + [拼接多轮任务](/tutorial/example_multi_turn.md)<br>+ [通用多轮任务](/tutorial/example_step_wise.md)<br>+ [调用智能体框架中的 ReAct 工作流](/tutorial/example_react.md)                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 数据流水线进阶能力 | + [Rollout 任务混合与选取](/tutorial/develop_selector.md)<br>+ [在线任务选择](https://github.com/modelscope/Trinity-RFT/tree/main/examples/bots) ([论文](https://arxiv.org/pdf/2510.26374))<br>+ [经验回放](https://github.com/modelscope/Trinity-RFT/tree/main/examples/ppo_countdown_exp_replay)<br>+ [高级数据处理能力 &  Human-in-the-loop](/tutorial/example_data_functionalities.md)                                                                                                                                                                                                                 |
| RL 算法开发/研究 | + [使用 Trinity-RFT 进行 RL 算法开发](/tutorial/example_mix_algo.md) ([论文](https://arxiv.org/pdf/2508.11408))<br>+ 不可验证的领域：[RULER](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k_ruler), [可训练 RULER](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k_trainable_ruler), [rubric-as-reward](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_rubric_as_reward) <br>+ [研究项目: group-relative REINFORCE](https://github.com/modelscope/Trinity-RFT/tree/main/examples/rec_gsm8k) ([论文](https://arxiv.org/abs/2509.24203)) |
| 深入认识 Trinity-RFT | + [完整配置指南](/tutorial/trinity_configs.md)<br>+ [用于快速验证和实验的 Benchmark 工具](https://github.com/modelscope/Trinity-RFT/tree/main/benchmark/README.md)<br>+ [理解 explorer-trainer 同步逻辑](/tutorial/synchronizer.md)                                                                                                                                                                                                                                                                                                                                                                     |



## 致谢


本项目基于许多优秀的开源项目构建，包括：

+ [verl](https://github.com/volcengine/verl)，[FSDP](https://pytorch.org/docs/stable/fsdp.html) 和 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 用于大模型训练；
+ [vLLM](https://github.com/vllm-project/vllm) 用于大模型推理；
+ [Data-Juicer](https://github.com/modelscope/data-juicer?tab=readme-ov-file) 用于数据处理流水线；
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
