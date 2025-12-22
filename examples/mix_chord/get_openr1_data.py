"""
Below we provide a script for processing the dataset into our required format for OpenR1 math tasks. Before running the dataset processing script, you need to fill in the tokenizer path in the script for filtering SFT data that is too long.
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
"""

import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# import re

# Set random seed for reproducibility
random.seed(42)

# Configuration parameters
TOKENIZER_MODEL_PATH = "YOUR MODEL TOKENIZER PATH"
MAX_TOKEN_LENGTH = 8196
SFT_SAMPLE_SIZE = 5000
RL_SAMPLE_SIZE = 20000
SYSTEM_PROMPT = """You are a helpful assistant that solves MATH problems. You should first think about the reasoning process in mind and then provides the user with the answer. You should present your reasoning process using the format: <think>\n ...your reasoning process here... </think>\n first. You should always include your final answer in \\boxed{} as closed-form results."""


def can_convert_to_int(answer):
    """Check if answer can be directly converted to integer"""
    try:
        int(answer)
        return True
    except (ValueError, TypeError):
        return False


def contains_chinese(text):
    """There are many incorrect translated problems in OpenR1. We may want to filter them out."""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
        if "\u3400" <= char <= "\u4dbf":
            return True
    return False


def process_dataset(openr1_ds, tokenizer):
    """Process dataset and filter out instances that don't meet criteria"""
    processed_data = []

    for instance in tqdm(openr1_ds, desc="Processing dataset"):
        # Filter out answers that cannot be directly converted to int
        # if not can_convert_to_int(instance["answer"]):
        #     continue

        if contains_chinese(instance["problem"]):
            continue

        # Process generations
        generations_keeped = []
        correctness_list = instance.get("correctness_math_verify", [])
        generations = instance.get("generations", [])

        for i, generation in enumerate(generations):
            # Check correctness_math_verify
            if i >= len(correctness_list) or not correctness_list[i]:
                continue

            # Check token length
            tokenized_length = len(tokenizer.tokenize(generation))
            if tokenized_length > MAX_TOKEN_LENGTH:
                continue

            generations_keeped.append(generation)

        # Add to processed data if there are kept generations
        if generations_keeped:
            processed_data.append(
                {
                    "problem": instance["problem"],
                    "answer": instance["answer"],
                    "generations": generations_keeped,
                }
            )

    return processed_data


def create_sft_dataset(data, sample_size):
    """Create SFT dataset with message format"""
    sft_messages = []

    # Random sample specified number of instances
    sampled_data = random.sample(data, min(sample_size, len(data)))

    for instance in sampled_data:
        # Randomly select one generation as response
        generation = random.choice(instance["generations"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instance["problem"]},
            {"role": "assistant", "content": generation},
        ]
        sft_messages.append({"messages": messages})

    return sft_messages, sampled_data


def create_rl_dataset(data, sample_size, output_dir):
    """Create RL dataset in HuggingFace format"""

    filtered_data = [d for d in data if can_convert_to_int(d["answer"])]
    print("Number of instances can convert to int: ", len(filtered_data))

    # Filter instances with at least 2 generations
    filtered_data = [d for d in filtered_data if len(d["generations"]) >= 2]
    print(f"Number of instances with >= 2 generations: {len(filtered_data)}")

    # or No filter
    # filtered_data = data

    # Random sample
    sampled_data = random.sample(filtered_data, min(sample_size, len(filtered_data)))

    # Prepare data in HuggingFace format (only problem and answer)
    rl_data = []
    for instance in sampled_data:
        rl_data.append({"problem": instance["problem"], "answer": instance["answer"]})

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSONL format for HuggingFace datasets
    train_file = os.path.join(output_dir, "train.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in rl_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Create dataset_dict.json for HuggingFace format
    dataset_info = {
        "citation": "",
        "description": "OpenR1 RLVR dataset subset",
        "splits": {"train": {"name": "train", "num_examples": len(rl_data)}},
    }

    with open(os.path.join(output_dir, "dataset_dict.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Saved RL dataset to {output_dir}")
    print(f"Total instances: {len(rl_data)}")

    return sampled_data  # Return sampled data with generations for reference


def save_json(data, filename):
    """Save data to JSON file"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(data)} instances to {filename}")


def main():
    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    openr1_ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train").to_list()

    print(f"Original dataset size: {len(openr1_ds)}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH, use_fast=False)

    # Process dataset with filtering
    print("Processing dataset with filters...")
    processed_data = process_dataset(openr1_ds, tokenizer)
    print(f"Processed dataset size: {len(processed_data)}")

    # Create SFT dataset
    print(f"\nCreating SFT dataset (sampling {SFT_SAMPLE_SIZE} instances)...")
    sft_dataset, sampled_for_sft = create_sft_dataset(processed_data, SFT_SAMPLE_SIZE)
    save_json(sft_dataset, "openr1_sft_dataset.json")

    # Create RL dataset from remaining data
    print("\nCreating RL dataset...")
    # Remove instances already used for SFT
    remaining_data = [d for d in processed_data if d not in sampled_for_sft]
    print(f"Remaining data after SFT sampling: {len(remaining_data)}")

    # Create RL dataset in HuggingFace format
    rl_output_dir = "openr1_rl_dataset"
    sampled_rl_data = create_rl_dataset(remaining_data, RL_SAMPLE_SIZE, rl_output_dir)

    # Optionally save the full RL data with generations for reference
    # save_json(sampled_rl_data, "openr1_rl_dataset_with_generations.json")

    print("\n" + "=" * 50)
    print("Dataset generation completed!")
    print(f"SFT dataset: {len(sft_dataset)} instances")
    print(f"RL dataset: {len(sampled_rl_data)} instances")
    print("=" * 50)


if __name__ == "__main__":
    main()
