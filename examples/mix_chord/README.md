# Example: CHORD Algorithm

Below we show an example of implementing the [CHORD](https://arxiv.org/pdf/2508.11408) algorithm.

Here we provide a basic runnable example demonstrating the core functionality of CHORD. The hyperparameters used in our experiments may not be optimal across different datasets—we encourage researchers to build upon this implementation and explore further improvements.

If you are interested in implementing your own algorithm, you may refer to the [documentation](../../docs/sphinx_doc/source/tutorial/example_mix_algo.md) for guidance.

## How to run

### Install Trinity-RFT

First, you should install Trinity-RFT.

Please follow the guide in [README.md](../../README.md) to install the dependencies and set up the environment.

### Prepare the models and datasets

Then you should prepare the models and datasets, and fill them in the configuration file.

You should first download the model you want from Hugging Face or ModelScope, for example:
```bash
# Using Hugging Face
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $MODEL_PATH/Qwen/Qwen2.5-1.5B-Instruct

# Using ModelScope
modelscope download {model_name} --local_dir $MODEL_PATH/{model_name}
```

For the dataset, you need to prepare both the SFT dataset and the RL dataset. Below we provide a script for processing the dataset into our required format.

Before running the dataset processing script, you need to fill in the tokenizer path in the script for filtering SFT data that is too long.
You can also change the sample size if you want.
```python
TOKENIZER_MODEL_PATH = "YOUR MODEL TOKENIZER PATH"
MAX_TOKEN_LENGTH = 8196
SFT_SAMPLE_SIZE = 5000
PREFERENCE_SAMPLE_SIZE = 20000
```

Then just run the script:
```bash
python examples/mix_chord/get_openr1_data.py
```
This may take a while to run.

> **Note**: Here we provide scripts for sampling SFT and RL data from the OpenR1 dataset, but unfortunately, since our original experiments did not use a fixed random seed, the data selection and ordering may differ from the paper.

### Modify the running script

Fill in the config file in [`mix_chord.yaml`](mix_chord.yaml).

### Run the script

```bash
# Stop existing ray processes
ray stop

# Start ray
ray start --head

# Run Trinity
trinity run --config examples/mix_chord/mix_chord.yaml
```

## Citation

If you find this code useful, please consider citing our paper:
```bibtex
@misc{TrinityRFT,
      title={Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models},
      author={Xuchen Pan and Yanxi Chen and Yushuo Chen and Yuchang Sun and Daoyuan Chen and Wenhao Zhang and Yuexiang Xie and Yilun Huang and Yilei Zhang and Dawei Gao and Weijie Shi and Yaliang Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2505.17826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17826},
}

@misc{MIXCHORD,
      title={On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting},
      author={Wenhao Zhang and Yuexiang Xie and Yuchang Sun and Yanxi Chen and Guoyin Wang and Yaliang Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2508.11408},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.11408},
}
```
