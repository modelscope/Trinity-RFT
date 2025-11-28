## 💡 What is Trinity-RFT?


Trinity-RFT is a general-purpose, flexible and user-friendly framework for LLM reinforcement fine-tuning (RFT).
It decouples RFT into three components that work in coordination:

* **Explorer** generates experience data via agent-environment interaction;

* **Trainer** updates model weights by minimizing losses on the data;

* **Buffer** pipelines data processing throughout the RFT lifecycle.


Trinity-RFT provides functionalities for users with different backgrounds and objectives:

* 🤖 **Agent application developers:** Train LLM-powered agents and improve their capabilities in specific domains [[tutorial]](/tutorial/develop_workflow.md)

* 🧠 **Reinforcement learning researchers:** Design, implement and validate new RL algorithms using compact, plug-and-play modules that allow non-invasive customization [[tutorial]](/tutorial/develop_algorithm.md)

* 📊 **Data engineers:** Create RFT datasets and build data pipelines for cleaning, augmentation, and human-in-the-loop scenarios [[tutorial]](/tutorial/develop_operator.md)




## 🔨 Tutorials and Guidelines


| Category | Tutorial / Guideline      |
| --- | ----|
| *Run diverse RFT modes* | + [Quick start: GRPO on GSM8k](/tutorial/example_reasoning_basic.md)<br>+ [Off-policy RFT](/tutorial/example_reasoning_advanced.md)<br>+ [Fully asynchronous RFT](/tutorial/example_async_mode.md)<br>+ [Offline learning by DPO or SFT](/tutorial/example_dpo.md)     |
| *Multi-step agentic RL* | + [Concatenated multi-turn workflow](/tutorial/example_multi_turn.md)<br>+ [General multi-step workflow](/tutorial/example_step_wise.md)<br>+ [ReAct workflow with an agent framework](/tutorial/example_react.md)  <br>+ [Example: train a web-search agent](https://github.com/modelscope/Trinity-RFT/tree/main/examples/agentscope_websearch) |
| *Full-lifecycle data pipelines* | + [Rollout task mixing and selection](/tutorial/develop_selector.md)<br>+ [Online task curriculum](https://github.com/modelscope/Trinity-RFT/tree/main/examples/bots) (📝 [paper](https://arxiv.org/pdf/2510.26374))<br>+ [Research project: learn-to-ask](https://github.com/modelscope/Trinity-RFT/tree/main/examples/learn_to_ask) (📝 [paper](https://arxiv.org/pdf/2510.25441)) <br>+ [Experience replay with prioritization](https://github.com/modelscope/Trinity-RFT/tree/main/examples/ppo_countdown_exp_replay)<br>+ [Advanced data processing & human-in-the-loop](/tutorial/example_data_functionalities.md)  |
| *Algorithm development* | + [RL algorithm development with Trinity-RFT](/tutorial/example_mix_algo.md) (📝 [paper](https://arxiv.org/pdf/2508.11408))<br>+ [Research project: group-relative REINFORCE](https://github.com/modelscope/Trinity-RFT/tree/main/examples/rec_gsm8k) (📝 [paper](https://arxiv.org/abs/2509.24203)) <br>+ Non-verifiable domains: [RULER](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k_ruler), [trainable RULER](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_gsm8k_trainable_ruler), [rubric-as-reward](https://github.com/modelscope/Trinity-RFT/tree/main/examples/grpo_rubric_as_reward) |
| *Going deeper into Trinity-RFT* | + [Full configurations](/tutorial/trinity_configs.md)<br>+ [Benchmark toolkit for quick verification and experimentation](https://github.com/modelscope/Trinity-RFT/tree/main/benchmark/README.md)<br>+ [Understand the coordination between explorer and trainer](/tutorial/synchronizer.md)    |




## 🌟 Key Features

* **Flexible RFT Modes:**
  - Supports synchronous/asynchronous, on-policy/off-policy, and online/offline RL.
  - Rollout and training can run separately and scale independently across devices.
  - Boost sample and time efficiency by experience replay.

  <img src="https://img.alicdn.com/imgextra/i3/O1CN01E7NskS1FFoTI9jlaQ_!!6000000000458-2-tps-1458-682.png" alt="RFT modes supported by Trinity-RFT" width="600" />

* **Agentic RL Support:**
  - Supports both concatenated and general multi-step agentic workflows.
  - Able to directly train agent applications developed using agent frameworks like [AgentScope](https://github.com/agentscope-ai/agentscope).

  <img src="https://img.alicdn.com/imgextra/i1/O1CN01z1i7kk1jlMEVa8ZHV_!!6000000004588-2-tps-1262-695.png" alt="Agentic workflows" width="600" />

* **Full-Lifecycle Data Pipelines:**
  - Enables pipeline processing of rollout tasks and experience samples.
  - Active data management (prioritization, cleaning, augmentation, etc.) throughout the RFT lifecycle.
  - Native support for multi-task joint learning and online task curriculum construction.

  <img src="https://img.alicdn.com/imgextra/i2/O1CN01Gk9CRw28NsL09nbOj_!!6000000007921-2-tps-2530-660.png" alt="Data pipeline design" width="720" />

* **User-Friendly Design:**
  - Plug-and-play modules and decoupled architecture, facilitating easy adoption and development.
  - Rich graphical user interfaces enable low-code usage.

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
