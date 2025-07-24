"""
We use this script to create the huggingface format dataset files for the alfworld dataset.
NOTE: You need to install the alfworld dataset in first: https://github.com/alfworld/alfworld
"""
import glob
import json
import os
import random

random.seed(42)


def create_dataset_files(output_dir, train_size=1024, test_size=100):
    # The ALFWORLD_DATA is the dataset path in the environment variable ALFWORLD_DATA, you need to set it when install alfworld dataset
    from alfworld.info import ALFWORLD_DATA

    # get all matched game files
    game_files = glob.glob(os.path.expanduser(f"{ALFWORLD_DATA}/json_2.1.1/train/*/*/game.tw-pddl"))

    # get absolute path
    game_files = [os.path.abspath(file) for file in game_files]
    game_files = sorted(game_files)

    # randomly sellect the game files
    sellected_game_files = random.sample(game_files, train_size + test_size)

    # make the output directory
    os.makedirs(output_dir, exist_ok=True)

    # for webshop dataset, we just need the session id as the task id
    all_data = []
    for game_file_path in sellected_game_files:
        all_data.append({"game_file": game_file_path, "target": ""})

    # split the train and test data
    train_data = all_data[:train_size]
    test_data = all_data[train_size : train_size + test_size]

    # create dataset_dict
    dataset_dict = {"train": train_data, "test": test_data}

    for split, data in dataset_dict.items():
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    # create dataset_dict.json
    dataset_info = {
        "citation": "",
        "description": "Custom dataset",
        "splits": {
            "train": {"name": "train", "num_examples": len(train_data)},
            "test": {"name": "test", "num_examples": len(test_data)},
        },
    }

    with open(os.path.join(output_dir, "dataset_dict.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = f"{current_file_dir}/alfworld_data"
    create_dataset_files(output_dir, train_size=1024, test_size=100)
