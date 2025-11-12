#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert BrowseComp-Plus dataset to Trinity-RFT format with train/test split.

This script converts the BrowseComp-Plus decrypted JSONL dataset into the format
expected by Trinity-RFT for RL training and evaluation.

Usage:
    python get_browser_comp_data_for_trinity.py --input path/to/browsecomp_plus_decrypted.jsonl --output_dir data/trinity_format

Environment Variables:
    BROWSECOMP_PATH: Path to BrowseComp-Plus directory (optional if --browsecomp_path is provided)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List


def load_browsecomp_data(jsonl_path: Path) -> List[Dict]:
    """Load BrowseComp-Plus dataset from JSONL file.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of data samples
    """
    data = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
    return data


def convert_to_trinity_format(sample: Dict) -> Dict:
    """Convert a single BrowseComp-Plus sample to Trinity-RFT format.

    Args:
        sample: BrowseComp-Plus sample with fields: query_id, query, answer, gold_docs

    Returns:
        Trinity-RFT formatted sample with fields: query_id, query, answer
    """
    return {
        "query_id": sample["query_id"],
        "query": sample["query"],
        "answer": sample["answer"],
    }


def save_jsonl(data: List[Dict], output_path: Path):
    """Save data to JSONL file.

    Args:
        data: List of data samples
        output_path: Path to save the JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BrowseComp-Plus dataset to Trinity-RFT format with train/test split"
    )
    parser.add_argument(
        "--input",
        help="Path to BrowseComp-Plus decrypted JSONL file. If not provided, will look for "
        "browsecomp_plus_decrypted.jsonl in BrowseComp-Plus data directory",
    )
    parser.add_argument(
        "--output_dir",
        default="data/trinity_format",
        help="Output directory for Trinity-RFT formatted data (default: %(default)s)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=400,
        help="Number of samples for training (default: %(default)s)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=200,
        help="Number of samples for testing (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--browsecomp_path",
        help="Path to BrowseComp-Plus directory. If not provided, will use BROWSECOMP_PATH env variable",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Determine BrowseComp-Plus path
    browsecomp_path = args.browsecomp_path or os.environ.get("BROWSECOMP_PATH")

    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        if not browsecomp_path:
            print("Error: Please provide --input path or set BROWSECOMP_PATH environment variable")
            sys.exit(1)
        input_path = Path(browsecomp_path) / "data" / "browsecomp_plus_decrypted.jsonl"

    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("\nPlease make sure you have:")
        print("1. Downloaded BrowseComp-Plus dataset")
        print("2. Decrypted the dataset using the decryption script")
        print("3. Set the correct path using --input or BROWSECOMP_PATH")
        sys.exit(1)

    print(f"Loading data from {input_path}")
    data = load_browsecomp_data(input_path)
    print(f"Loaded {len(data)} samples")

    # Shuffle data with fixed seed
    random.shuffle(data)

    # Split into train and test
    total_needed = args.train_size + args.test_size
    if len(data) < total_needed:
        print(
            f"Warning: Dataset has {len(data)} samples, but {total_needed} requested. "
            f"Adjusting split proportionally."
        )
        train_size = int(len(data) * args.train_size / total_needed)
        test_size = len(data) - train_size
    else:
        train_size = args.train_size
        test_size = args.test_size

    train_data = data[:train_size]
    test_data = data[train_size : train_size + test_size]

    print("\nSplitting data:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    # Convert to Trinity format
    train_trinity = [convert_to_trinity_format(s) for s in train_data]
    test_trinity = [convert_to_trinity_format(s) for s in test_data]

    # Save converted data
    save_jsonl(train_trinity, output_dir / "train.jsonl")
    save_jsonl(test_trinity, output_dir / "test.jsonl")

    print("\nConversion complete!")
    print(f"Random seed used: {args.seed}")
    print("\nNext steps:")
    print(f"1. Set environment variable: export TRINITY_TASKSET_PATH={output_dir.absolute()}")
    print("2. Make sure BROWSECOMP_PATH environment variable is set")
    print(
        "3. Run training with: python -m trinity.train --config examples/browse_comp_plus/bcp_config.yaml"
    )


if __name__ == "__main__":
    main()
