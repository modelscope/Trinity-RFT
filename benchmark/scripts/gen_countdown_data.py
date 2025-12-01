"""
Modified from https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import argparse
import json
import os

from datasets import load_dataset
from verl.utils.hdfs_io import copy, makedirs

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "countdown"
)


def make_prefix(dp):
    target = dp["target"]
    numbers = dp["nums"]
    system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer."""
    task_desc = f"""User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>"""
    final_prompt = f"{system_prompt}\n{task_desc}"
    return final_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_size", type=int, default=320000)
    parser.add_argument("--test_size", type=int, default=7680)

    args = parser.parse_args()

    data_source = "countdown"
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def process_fn(example, idx):
        question = make_prefix(example)
        data = {
            "question": question,
            "answer": json.dumps(
                {
                    "numbers": example["nums"],
                    "target": example["target"],
                }
            ),
        }
        return data

    train_dataset = train_dataset.map(
        function=process_fn, with_indices=True, remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        function=process_fn, with_indices=True, remove_columns=test_dataset.column_names
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_json(os.path.join(local_dir, "train.jsonl"))
    test_dataset.to_json(os.path.join(local_dir, "test.jsonl"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
