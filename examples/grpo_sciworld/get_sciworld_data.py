"""
We use this script to create the huggingface format dataset files for the sciworld dataset.
NOTE: You need to install the ScienceWorld dataset first: https://github.com/allenai/ScienceWorld
"""
import json
import os
import random

random.seed(42)

task_variations = {
    "boil": 30,
    "melt": 30,
    "freeze": 30,
    "change-the-state-of-matter-of": 30,
    "use-thermometer": 540,
    "measure-melting-point-known-substance": 436,
    "measure-melting-point-unknown-substance": 300,
    "power-component": 20,
    "power-component-renewable-vs-nonrenewable-energy": 20,
    "test-conductivity": 900,
    "test-conductivity-of-unknown-substances": 600,
    "find-living-thing": 300,
    "find-non-living-thing": 300,
    "find-plant": 300,
    "find-animal": 300,
    "grow-plant": 126,
    "grow-fruit": 126,
    "chemistry-mix": 32,
    "chemistry-mix-paint-secondary-color": 36,
    "chemistry-mix-paint-tertiary-color": 36,
    "lifespan-longest-lived": 125,
    "lifespan-shortest-lived": 125,
    "lifespan-longest-lived-then-shortest-lived": 125,
    "identify-life-stages-1": 14,
    "identify-life-stages-2": 10,
    "inclined-plane-determine-angle": 168,
    "inclined-plane-friction-named-surfaces": 1386,
    "inclined-plane-friction-unnamed-surfaces": 162,
    "mendelian-genetics-known-plant": 120,
    "mendelian-genetics-unknown-plant": 480,
}


def create_dataset_files(output_dir, train_task_names, test_task_names, jar_path, percentage=0.6):
    # make the output directory
    os.makedirs(output_dir, exist_ok=True)

    train_data = []
    test_data = []

    for task_name in train_task_names:
        total_var = task_variations[task_name]
        for i in range(int(total_var * percentage)):
            task_config = {
                "task_name": task_name,
                "var_num": i,
                "jar_path": jar_path,
            }
            task_desc = json.dumps(task_config)
            train_data.append({"task_desc": task_desc, "targe": ""})

    random.shuffle(train_data)

    for task_name in test_task_names:
        total_var = task_variations[task_name]
        for i in range(int(total_var * percentage)):
            task_config = {
                "task_name": task_name,
                "var_num": i,
                "jar_path": jar_path,
            }
            task_desc = json.dumps(task_config)
            test_data.append({"task_desc": task_desc, "targe": ""})

    random.shuffle(test_data)

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
    # NOTE: Mannually set the jar path here.
    jar_path = "/your/path/ScienceWorld/scienceworld/scienceworld.jar"
    # Check if the jar file exists, raise an error if it doesn't exist.
    if not os.path.exists(jar_path):
        raise FileNotFoundError(
            f"JAR file not found at {jar_path}, please set the jar path mannually."
        )

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = f"{current_file_dir}/sciworld_data"
    train_task_names = [
        "boil",
        "melt",
        "change-the-state-of-matter-of",
        "use-thermometer",
        "measure-melting-point-known-substance",
        "power-component",
        "test-conductivity",
        "find-living-thing",
        "find-plant",
        "grow-plant",
        "chemistry-mix",
        "chemistry-mix-paint-secondary-color",
        "lifespan-shortest-lived",
        "identify-life-stages-2",
        "inclined-plane-determine-angle",
        "inclined-plane-friction-named-surfaces",
        "mendelian-genetics-known-plant",
    ]
    test_task_names = list(task_variations.keys() - set(train_task_names))
    create_dataset_files(output_dir, train_task_names, test_task_names, jar_path, percentage=0.5)
