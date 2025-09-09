# Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models


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

* **Train agent applications with RL and minimal migration cost**
  - Implement agent-environment interaction logic in a single workflow class ([example](/tutorial/example_multi_turn.md)),
  - Or import workflows from agent frameworks like AgentScope ([example](/tutorial/example_react.md)).

* **Rapid RL algorithm design and validation**
  - Develop custom RL algorithms (loss design, sampling strategy, etc.) in compact, plug-and-play classes ([example](/tutorial/example_mix_algo.md)).

* **Custom datasets and data pipelines for RFT**
  - Design task-specific datasets and build data pipelines for cleaning, augmentation, and human-in-the-loop scenarios ([example](/tutorial/example_data_functionalities.md)).


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
