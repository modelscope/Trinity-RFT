"""
We use this script to create the huggingface format dataset files for the webshop dataset.
"""
import json
import os


def create_dataset_files(output_dir, train_size=4096, test_size=100):
    # make the output directory
    os.makedirs(output_dir, exist_ok=True)

    # for webshop dataset, we just need the session id as the task id
    all_data = []
    for task_id in range(train_size + test_size):
        all_data.append({"task_id": task_id, "target": ""})

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
    output_dir = f"{current_file_dir}/webshop_data"
    create_dataset_files(output_dir, train_size=4096, test_size=100)
