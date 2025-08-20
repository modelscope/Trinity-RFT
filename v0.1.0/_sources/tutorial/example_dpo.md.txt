# Offline DPO

This example describes DPO based on the Qwen-2.5-1.5B-Instruct model and [Human-like-DPO-dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset).

## Step 1: Model and Data Preparation

### Model Preparation

Download the Qwen-2.5-1.5B-Instruct model to the local directory `$MODEL_PATH/Qwen2.5-1.5B-Instruct`:

```shell
# Using Modelscope
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir $MODEL_PATH/Qwen2.5-1.5B-Instruct

# Using Huggingface
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $MODEL_PATH/Qwen2.5-1.5B-Instruct
```

More details of model downloading are referred to [ModelScope](https://modelscope.cn/docs/models/download) or [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli).

### Data Preparation

Download the Human-Like-DPO-Dataset dataset to the local directory `$DATASET_PATH/human_like_dpo_dataset`:

```shell
# Using Modelscope
modelscope download --dataset HumanLLMs/Human-Like-DPO-Dataset --local_dir $DATASET_PATH/human_like_dpo_dataset

# Using Huggingface
huggingface-cli download HumanLLMs/Human-Like-DPO-Dataset --repo-type dataset --local-dir $DATASET_PATH/human_like_dpo_dataset
```

More details of dataset downloading are referred to [ModelScope](https://modelscope.cn/docs/datasets/download) or [Huggingface](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#download-a-dataset-or-a-space).

Note that the dataset has the keys `prompt`, `chosen` and `rejected`. If not, pass the proper keys to the config.

## Step 2: Setup Configuration and Run Experiment

### Configuration

We use the configurations in [`dpo.yaml`](https://github.com/modelscope/Trinity-RFT/tree/main/examples/dpo_humanlike/dpo.yaml) and [`train_dpo.yaml`](https://github.com/modelscope/Trinity-RFT/tree/main/examples/dpo_humanlike/train_dpo.yaml) for this experiment. Some important setups are listed in the following:

We run the experiment in a train mode, as there is no Explorer. To enable this mode, we config `mode` to `train` and pass the data path to the trainer.

```yaml
project: <project_name>
name: <experiment_name>
mode: train
algorithm:
  algorithm_type: dpo
checkpoint_root_dir: /PATH/TO/CHECKPOINT/
model:
  model_path: /PATH/TO/MODEL/
cluster:
  node_num: 1
  gpu_per_node: 8
buffer:
  total_epochs: 2
  batch_size: 64
  trainer_input:
    experience_buffer:
      name: dpo_buffer
      storage_type: file
      path: /PATH/TO/DATASET/
      format:
        prompt_type: plaintext # plaintext/messages/chatpair
        prompt_key: prompt
        chosen_key: chosen
        rejected_key: rejected
trainer:
  trainer_config_path: 'examples/dpo_humanlike/train_dpo.yaml'
  save_interval: 30
  actor_use_kl_loss: True
  actor_kl_loss_coef: 0.1  # value of beta in DPO
```

### Run the Experiment

Run RFT process with the following command:

```shell
trinity run --config examples/dpo_humanlike/dpo.yaml
```
