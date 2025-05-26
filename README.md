

<!-- ![trinity-rft](./docs/sphinx_doc/assets/trinity-title.png) -->

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01lvLpfw25Pl4ohGZnU_!!6000000007519-2-tps-1628-490.png" alt="Trinity-RFT" style="height: 120px;">
</div>


[![paper](http://img.shields.io/badge/cs.LG-2505.17826-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.17826)
[![doc](https://img.shields.io/badge/Docs-blue?logo=markdown)](https://modelscope.github.io/Trinity-RFT/)
[![pypi](https://img.shields.io/pypi/v/trinity-rft?logo=pypi&color=026cad)](https://pypi.org/project/trinity-rft/0.1.0/)
![license](https://img.shields.io/badge/license-Apache--2.0-000000.svg)



**Trinity-RFT is a general-purpose, flexible, scalable and user-friendly framework designed for reinforcement fine-tuning (RFT) of large language models (LLM).**


Built with a decoupled design, seamless integration for agent-environment interaction, and systematic data processing pipelines, Trinity-RFT can be easily adapted for diverse application scenarios, and serve as a unified platform for exploring advanced reinforcement learning (RL) paradigms.





## Vision of this project


Current RFT approaches, such as RLHF (Reinforcement Learning from Human Feedback) with proxy reward models or training long-CoT reasoning models with rule-based rewards, are limited in their ability to handle dynamic, real-world, and continuous learning.

Trinity-RFT envisions a future where AI agents learn by interacting directly with environments, collecting delayed or complex reward signals, and continuously refining their behavior through RL.


For example, imagine an AI scientist that designs an experiment, executes it, waits for feedback (while working on other tasks concurrently), and iteratively updates itself based on true environmental rewards when the experiment is finally finished.


Trinity-RFT offers a path into this future by providing various useful features.





## Key features



+ **Unified RFT modes & algorithm support.**
Trinity-RFT unifies and generalizes existing RFT methodologies into a flexible and configurable framework, supporting synchronous/asynchronous, on-policy/off-policy, and online/offline training, as well as hybrid modes that combine them seamlessly into a single learning process.


+ **Agent-environment interaction as a first-class citizen.**
Trinity-RFT allows delayed rewards in multi-step/time-lagged feedback loops, handles long-tailed latencies and environment/agent failures gracefully, and supports distributed deployment where explorers and trainers can operate across separate devices and scale up independently.



+ **Data processing pipelines optimized for RFT with diverse/messy data.**
These include converting raw datasets to task sets for RL, cleaning/filtering/prioritizing experiences stored in the replay buffer, synthesizing data for tasks and experiences, offering user interfaces for human in the loop, etc.



## The design of Trinity-RFT


<!-- ![design](./docs/sphinx_doc/assets/trinity-design.png) -->

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i2/O1CN01X5jFm81peNsADtRt2_!!6000000005385-2-tps-3298-1498.png" alt="Trinity-RFT">
</div>





The overall design of Trinity-RFT exhibits a trinity:
+ RFT-core;
+ agent-environment interaction;
+ data processing pipelines;

and the design of RFT-core also exhibits a trinity:
+ explorer;
+ trainer;
+ buffer.



The *explorer*, powered by the rollout model, interacts with the environment and generates rollout trajectories to be stored in the experience buffer.

The *trainer*, powered by the policy model, samples batches of experiences from the buffer and updates the policy model via RL algorithms.

These two can be completely decoupled and act asynchronously on separate machines, except that they share the same experience buffer, and their model weights are synchronized once in a while.
Such a decoupled design is crucial for making the aforementioned features of Trinity-RFT possible.

<!-- e.g., flexible and configurable RFT modes (on-policy/off-policy, synchronous/asynchronous, immediate/lagged rewards),
fault tolerance for failures of explorer (agent/environment) or trainer,
high efficiency in the presence of long-tailed rollout latencies,
data processing pipelines and human in the loop of RFT (e.g., via acting on the experience buffer, which is implemented as a persistent database),
among others. -->



Meanwhile, Trinity-RFT has done a lot of work to ensure high efficiency and robustness in every component of the framework,
e.g., utilizing NCCL (when feasible) for model weight synchronization, sequence concatenation with proper masking for multi-turn conversations and ReAct-style workflows, pipeline parallelism for the synchronous RFT mode,
asynchronous and concurrent LLM inference for rollout,
fault tolerance for agent/environment failures,
among many others.



## Getting started


> [!NOTE]
> This project is currently under active development. Comments and suggestions are welcome!




### Step 1: preparations




Installation from source (recommended):

```shell
# Pull the source code from GitHub
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# Create a new environment using Conda or venv
# Option 1: Conda
conda create -n trinity python=3.10
conda activate trinity

# Option 2: venv
python3.10 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode
# for bash
pip install -e .[dev]
# for zsh
pip install -e .\[dev\]

# Install flash-attn after all dependencies are installed
# Note: flash-attn will take a long time to compile, please be patient.
pip install flash-attn -v
# Try the following command if you encounter errors during installation
# pip install flash-attn -v --no-build-isolation
```

Installation using pip:

```shell
pip install trinity-rft==0.1.0
```

Installation from docker:
we have provided a dockerfile for Trinity-RFT (trinity)

```shell
git clone https://github.com/modelscope/Trinity-RFT
cd Trinity-RFT

# build the docker image
# Note: you can edit the dockerfile to customize the environment
# e.g., use pip mirrors or set api key
docker build -f scripts/docker/Dockerfile -t trinity-rft:latest .

# run the docker image
docker run -it --gpus all --shm-size="64g" --rm -v $PWD:/workspace -v <root_path_of_data_and_checkpoints>:/data trinity-rft:latest
```


Trinity-RFT requires
Python version >= 3.10,
CUDA version >= 12.4,
and at least 2 GPUs.


### Step 2: prepare dataset and model


Trinity-RFT supports most datasets and models from Huggingface and ModelScope.


**Prepare the model** in the local directory `$MODEL_PATH/{model_name}`:

```bash
# Using Huggingface
huggingface-cli download {model_name} --local-dir $MODEL_PATH/{model_name}

# Using Modelscope
modelscope download {model_name} --local_dir $MODEL_PATH/{model_name}
```

For more details about model downloading, please refer to [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) or  [ModelScope](https://modelscope.cn/docs/models/download).



**Prepare the dataset** in the local directory `$DATASET_PATH/{dataset_name}`:

```bash
# Using Huggingface
huggingface-cli download {dataset_name} --repo-type dataset --local-dir $DATASET_PATH/{dataset_name}

# Using Modelscope
modelscope download --dataset {dataset_name} --local_dir $DATASET_PATH/{dataset_name}
```

For more details about dataset downloading, please refer to [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space) or [ModelScope](https://modelscope.cn/docs/datasets/download).



### Step 3: configurations


For convenience, Trinity-RFT provides a web interface for configuring your RFT process.

> [!NOTE]
> This is an experimental feature, and we will continue to improve it.

```bash
trinity studio --port 8080
```


Then you can configure your RFT process in the web page and generate a config file.
You can save the config for later use or run it directly as described in the following section.


Advanced users can also configure the RFT process by editing the config file directly.
We provide a set of example config files in [`examples`](examples/).


### Step 4: run the RFT process


First, start a ray cluster with the following command:

```shell
# On master node
ray start --head

# On worker nodes
ray start --address=<master_address>
```

Optionally, we can login into [wandb](https://docs.wandb.ai/quickstart/) to better monitor the RFT process:

```shell
export WANDB_API_KEY=<your_api_key>
wandb login
```

Then, for command-line users, run the RFT process with the following command:

```shell
trinity run --config <config_path>
```

> For example, below is the command for fine-tuning Qwen-2.5-1.5B-Instruct on GSM8k dataset using GRPO algorithm:
> ```shell
> trinity run --config examples/grpo_gsm8k/gsm8k.yaml
> ```

For studio users, just click the "Run" button in the web page.


For more detailed examples about how to use Trinity-RFT, please refer to the following tutorials:
+ [A quick example with GSM8k](./docs/sphinx_doc/source/tutorial/example_reasoning_basic.md)
+ [Off-policy mode of RFT](./docs/sphinx_doc/source/tutorial/example_reasoning_advanced.md)
+ [Asynchronous mode of RFT](./docs/sphinx_doc/source/tutorial/example_async_mode.md)
+ [Multi-turn tasks](./docs/sphinx_doc/source/tutorial/example_multi_turn.md)
+ [Offline learning by DPO](./docs/sphinx_doc/source/tutorial/example_dpo.md)
+ [Advanced data processing / human-in-the-loop](./docs/sphinx_doc/source/tutorial/example_data_functionalities.md)





## Advanced usage and full configurations


Please refer to [this document](./docs/sphinx_doc/source/tutorial/trinity_configs.md).





## Programming guide for developers


Please refer to [this document](./docs/sphinx_doc/source/tutorial/trinity_programming_guide.md).


## Upcoming features

A tentative roadmap: https://github.com/modelscope/Trinity-RFT/issues/51



## Contribution guide


This project is currently under active development, and we welcome contributions from the community!


Code style check:

```shell
pre-commit run --all-files
```



Unit tests:

```shell
python -m pytest tests
```



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
```plain
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
