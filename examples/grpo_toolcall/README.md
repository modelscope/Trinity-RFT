# GRPO on ToolAce dataset

This example shows the usage of GRPO on the [ToolAce](https://huggingface.co/datasets/Team-ACE/ToolACE) dataset.

We reference code from [Tool-N1](https://github.com/NVlabs/Tool-N1) for the data preprocessing script and the workflow construction.

The config files are located in [`toolace.yaml`](toolace.yaml) and [`train_toolace.yaml`](train_toolace.yaml).


## How to run
To preprocess the data into the format required our `toolcall_workflow`, run the following command: `python scripts/data_prepare/get_toolace_data.py`.

Then fill in the config file `toolace.yaml` and run the following command: `trinity run --config examples/grpo_toolcall/toolace.yaml`.

## Reward curve results

![](../../docs/sphinx_doc/assets/toolace_reward_curve.png)
