"""
Modified from https://github.com/rllm-org/rllm/blob/main/examples/frozenlake/prepare_frozenlake_data.py
"""
import os

import numpy as np
import pandas as pd

from trinity.common.constants import TASKSET_PATH_ENV_VAR

path_from_env = os.environ.get(TASKSET_PATH_ENV_VAR)
if path_from_env is not None:
    DATA_ROOT_DIR = os.path.dirname(path_from_env)
else:
    DATA_ROOT_DIR = os.path.join(os.path.dirname(__file__), "data")


def save_dataset_to_local(name: str, data: list[dict], split: str = "default") -> str:
    """Save dataset directly to local DATA_PATH.

    Args:
        name: Name of the dataset
        data: List of dictionaries containing the dataset examples
        split: Split name (e.g., 'train', 'test', 'default')

    Returns:
        str: Path to the saved parquet file
    """
    dataset_dir = os.path.join(DATA_ROOT_DIR, name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Convert to DataFrame and save
    data_df = pd.DataFrame(data)
    dataset_path = os.path.join(dataset_dir, f"{split}.parquet")
    data_df.to_parquet(dataset_path)

    print(
        f"Saved dataset '{name}' split '{split}' with {len(data)} examples at {dataset_path}. Make sure to set the environment variable {TASKSET_PATH_ENV_VAR} to {DATA_ROOT_DIR}/{name}."
    )

    return dataset_path


def prepare_frozenlake_data(train_size=10000, test_size=100, map_max_size=6):
    """
    Prepare and save FrozenLake datasets for training and testing.

    Args:
        train_size (int): Number of training examples to generate
        test_size (int): Number of test examples to generate

    Returns:
        tuple: (train_data, test_data) - Lists of data dictionaries
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random parameters for train and test sets
    train_seeds = np.random.randint(0, 100000, size=train_size)
    test_seeds = np.random.randint(0, 100000, size=test_size)
    train_sizes = np.random.randint(2, map_max_size, size=train_size)
    test_sizes = np.random.randint(2, map_max_size, size=test_size)
    train_ps = np.random.uniform(0.6, 0.85, size=train_size)
    test_ps = np.random.uniform(0.6, 0.85, size=test_size)

    def frozenlake_process_fn(seed, size, p, idx):
        """Process function to create FrozenLake task instances."""
        return {"seed": seed, "size": size, "p": p, "index": idx, "uid": f"{seed}_{size}_{p}"}

    # Create train and test data
    train_data = [
        frozenlake_process_fn(seed, train_sizes[idx], train_ps[idx], idx)
        for idx, seed in enumerate(train_seeds)
    ]
    test_data = [
        frozenlake_process_fn(seed, test_sizes[idx], test_ps[idx], idx)
        for idx, seed in enumerate(test_seeds)
    ]

    # Save datasets directly to local DATA_PATH
    save_dataset_to_local("frozenlake", train_data, "train")
    save_dataset_to_local("frozenlake", test_data, "test")

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = prepare_frozenlake_data()
    print(f"Train dataset: {len(train_data)} examples")
    print(f"Test dataset: {len(test_data)} examples")
    print("Sample train example:", train_data[0])
    print("Sample test example:", test_data[0])
