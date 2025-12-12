import argparse
import json

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ref-eval-path", type=str, required=True)
    parser.add_argument("--ref-eval-key", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading original dataset from {args.data_path}...")
    original_data = pd.read_parquet(args.data_path)
    prompt2linenum = {}
    for i, d in enumerate(original_data["prompt"]):
        prompt2linenum[d[0]["content"]] = i
    eval_results = [0.0 for _ in range(len(original_data))]
    print(f"Loading reference evaluation results from {args.ref_eval_path}...")
    print(f"Results will be written to the original dataset at a new column {args.ref_eval_key}...")
    with open(args.ref_eval_path, "r") as f:
        for line in f:
            item = json.loads(line)
            eval_results[prompt2linenum[item["question"][0]["content"]]] = np.mean(item["rewards"])
    original_data[args.ref_eval_key] = eval_results
    print(f"Dataset overwritten at {args.data_path}...")
    original_data.to_parquet(args.data_path)


if __name__ == "__main__":
    main()
