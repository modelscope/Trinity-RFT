## 💡 What is Trinity-RFT?

Trinity-RFT is a flexible, general-purpose framework for reinforcement fine-tuning (RFT) of large language models (LLMs). It decouples the RFT process into three key components: **Explorer**, **Trainer**, and **Buffer**, and provides functionalities for users with different backgrounds and objectives:

* 🤖 For agent application developers. [[tutorial]](/tutorial/develop_workflow.md)
  - Train agent applications to improve their ability to complete tasks in specific environments.
  - Examples: [Multi-Turn Interaction](/tutorial/example_multi_turn.md), [ReAct Agent](/tutorial/example_react.md)

* 🧠 For RL algorithm researchers. [[tutorial]](/tutorial/develop_algorithm.md)
  - Design and validate new reinforcement learning algorithms using compact, plug-and-play modules.
  - Example: [Mixture of SFT and GRPO](/tutorial/example_mix_algo.md)

* 📊 For data engineers. [[tutorial]](/tutorial/develop_operator.md)
  - Create task-specific datasets and build data pipelines for cleaning, augmentation, and human-in-the-loop scenarios.
  - Example: [Data Processing](/tutorial/example_data_functionalities.md)


## 🌟 Key Features

* **Flexible RFT Modes:**
  - Supports synchronous/asynchronous, on-policy/off-policy, and online/offline training. Rollout and training can run separately and scale independently across devices.

  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="RFT modes supported by Trinity-RFT" width="600" />

* **General Agentic-RL Support:**
  - Supports both concatenated and general multi-turn agentic workflows. Able to directly train agent applications developed using agent frameworks like AgentScope.

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="Agentic workflows" width="600" />

* **Full Lifecycle Data Pipelines:**
  - Enables pipeline processing of rollout and experience data, supporting active management (prioritization, cleaning, augmentation) throughout the RFT lifecycle.

  <img src="https://img.alicdn.com/imgextra/i2/O1CN01BfeHp61sXSlGjH7zQ_!!6000000005776-2-tps-1734-473.png" alt="Data pipeline design" width="600" />

* **User-Friendly Design:**
  - Modular, decoupled architecture for easy adoption and development. Rich graphical user interfaces enable low-code usage.

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01Ti0o4320RywoAuyhN_!!6000000006847-2-tps-3840-2134.png" alt="System architecture" width="600" />


## Acknowledgements

This project is built upon many excellent open-source projects, including:

+ [verl](https://github.com/volcengine/verl) and [PyTorch's FSDP](https://pytorch.org/docs/stable/fsdp.html) for LLM training;
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
