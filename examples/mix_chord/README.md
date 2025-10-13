# Example: CHORD Algorithm

Below is an example of implementing the [CHORD](https://arxiv.org/pdf/2508.11408) algorithm.

Here we provide a basic runnable example demonstrating the core functionality of CHORD. The hyperparameters used in our experiments may not be optimal across different datasets—we encourage researchers to build upon this implementation and explore further improvements.

If you are interested in implementing your own algorithm, you may refer to the [documentation](../../docs/sphinx_doc/source/tutorial/example_mix_algo.md) for guidance.

## Steps to Run CHORD

Below we show how to run CHORD on the ToolAce Dataset.

### Install Trinity-RFT

First, you should install Trinity-RFT.

Please follow the guide in [README.md](../../README.md) to install the dependencies and set up the environment.

### Prepare the Models and Datasets

Then, you should prepare the models and datasets and specify them in the configuration file.

#### Prepare Model

To download the `llama3.2-3b-instruct` model, you can run the following command:

```bash
modelscope download --model LLM-Research/Llama-3.2-3B-Instruct --local_dir $MODEL_PATH/{model_name}
```

#### Prepare Dataset

You need to prepare both the SFT and RL datasets.

Following the setting in [Tool-N1](https://github.com/NVlabs/Tool-N1/), we process the ToolAce dataset in a similar fashion in [`get_toolace_data.py`](../grpo_toolcall/get_toolace_data.py).

The RL dataset we use is [`datajuicer/Trinity-ToolAce-RL-split`](https://huggingface.co/datasets/datajuicer/Trinity-ToolAce-RL-split), and the SFT dataset we use is [`datajuicer/Trinity-ToolAce-SFT-split`](https://huggingface.co/datasets/datajuicer/Trinity-ToolAce-SFT-split).


### Modify the Running Script

Update the configuration in [`mix_chord_toolace.yaml`](mix_chord_toolace.yaml).

### Run the Script

```bash
# Stop existing ray processes
ray stop

# Start ray
ray start --head

# Run Trinity
trinity run --config examples/mix_chord/mix_chord_toolace.yaml
```

It takes around 3 hours to run on 8 H20 GPUs.

After the run, you may also want to convert the checkpoint to a Hugging Face checkpoint.

```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from trinity.common.models.utils import load_fsdp_state_dict_from_verl_checkpoint

# The following variables are assumed to be predefined:
# model_path, checkpoint_root_dir, project, name
model = AutoModelForCausalLM.from_pretrained(model_path)
ckp_path = os.path.join(checkpoint_root_dir, project, name, "global_step_100", "actor")
state_dict = load_fsdp_state_dict_from_verl_checkpoint(ckp_path)
model.load_state_dict(state_dict)
output_dir = os.path.join(ckp_path, "huggingface")

def save_to_huggingface_checkpoint(state_dict: dict, output_dir: str):
    """Convert state dict to Hugging Face format and save it.

    Args:
        state_dict: The state dict loaded from the Verl checkpoint.
        output_dir: The directory to save the Hugging Face checkpoint.
    """
    import os
    import torch
    from transformers import PreTrainedModel

    os.makedirs(output_dir, exist_ok=True)

    # Convert state dict keys to Hugging Face format if needed
    hf_state_dict = {}
    for key, value in state_dict.items():
        # Add any key mapping logic here if needed
        # Example:
        # if key.startswith("model."):
        #     new_key = key.replace("model.", "")
        #     hf_state_dict[new_key] = value
        # else:
        #     hf_state_dict[key] = value
        hf_state_dict[key] = value
    torch.save(hf_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

save_to_huggingface_checkpoint(state_dict, output_dir)
```

## Evaluate the Trained Model on BFCL

### Install and Adapt BFCL

To evaluate the model on the Berkeley Function-Calling Leaderboard (BFCL), you first need to install it and then apply a patch to add support for our specific model format (used in [Tool-N1](https://github.com/NVlabs/Tool-N1/)).

First, clone the BFCL repository and navigate into it:
```bash
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/
```

The [patch](./eval_bfcl/bfcl_reason_support.patch) is created based on a specific version of the codebase. To ensure it applies cleanly, check out the exact commit:
```bash
git checkout cd9429ccf3d4d04156affe883c495b3b047e6b64
```

Copy the [`bfcl_qwen_reason_support.patch`](./eval_bfcl/bfcl_reason_support.patch) file into the `gorilla` directory you are currently in.

```bash
cp ../eval_bfcl/bfcl_reason_support.patch .
```

Then, apply the patch directly from the `gorilla` root directory:
```bash
# (Optional but recommended) Check if the patch can be applied without errors
git apply --check bfcl_reason_support.patch

# Apply the patch
git apply bfcl_reason_support.patch
```
If the `check` command produces no output, the patch can be applied successfully. The patch will modify files inside the `berkeley-function-call-leaderboard` subdirectory.


### Run BFCL Evaluation

To run the evaluation, use the following script.
```bash
echo "[INFO] Running bfcl generate..."
bfcl generate \
  --model your-qwen-reason-model \
  --test-category live_simple,live_multiple,live_parallel,live_parallel_multiple,simple,multiple,parallel,parallel_multiple \
  --backend vllm \
  --num-gpus 8 \
  --num-threads 8 \
  --gpu-memory-utilization 0.9 \
  --local-model-path "${hf_path}"


# bfcl evaluate
echo "[INFO] Running bfcl evaluate..."
bfcl evaluate \
  --model your-qwen-reason-model \
  --test-category live_simple,live_multiple,live_parallel,live_parallel_multiple,simple,multiple,parallel,parallel_multiple
```

### View BFCL Results

The corresponding BFCL evaluation results will be in the `score` subfolder.
To view the results as well as the average scores, you can simply run [`view_single_model_result.py`](./eval_bfcl/view_single_model_result.py):

```bash
python ./eval_bfcl/view_single_model_result.py PATH_TO_SUBFOLDER
```

## Citation

If you find this code useful, please consider citing our papers:
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
