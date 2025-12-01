import argparse
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "guru_math"
)


def process_fn(example, idx):
    data = {
        "question": example["prompt"][0]["content"],
        "ground_truth": example["reward_model"]["ground_truth"],
    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    downloaded_file_path = hf_hub_download(
        repo_id="LLM360/guru-RL-92k",
        filename="train/math__combined_54.4k.parquet",
        repo_type="dataset",
    )
    dataset = load_dataset("parquet", data_files=downloaded_file_path, split="train")
    new_dataset = dataset.map(
        function=process_fn, with_indices=True, remove_columns=dataset.column_names
    ).shuffle()
    os.makedirs(args.local_dir, exist_ok=True)
    new_dataset.to_json(os.path.join(args.local_dir, "train.jsonl"))
