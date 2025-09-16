# -*- coding: utf-8 -*-
"""
This script creates Hugging Face dataset-formatted files (train.jsonl, test.jsonl)
from the 'callanwu/WebWalkerQA' dataset for Reinforcement Learning (RL) training.

The script performs the following actions:
1. Loads the 'callanwu/WebWalkerQA' dataset from the Hugging Face Hub.
2. Splits the data into training and test set candidates based on 'difficulty_level' and 'lang'.
   - 'easy' and 'en' entries are candidates for the test set.
   - All other entries are candidates for the training set.
3. Randomly samples a specified number of items from the candidate pools.
4. Saves the processed data into 'train.jsonl' and 'test.jsonl' files.
5. Generates a 'dataset_dict.json' file describing the dataset's structure.
"""
import os
import json
import random
from datasets import load_dataset

def create_dataset_files(output_dir, train_sample_size=160, test_sample_size=16):
    """
    Loads, processes, and saves the WebWalkerQA dataset.

    Args:
        output_dir (str): The directory where the output files will be saved.
        train_sample_size (int): The maximum number of samples for the training set.
        test_sample_size (int): The maximum number of samples for the test set.
    """
    print("Starting dataset file creation...")

    # 1. Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' is ready.")

    # 2. Load the dataset from the Hugging Face Hub.
    print("Loading 'callanwu/WebWalkerQA' dataset from the Hugging Face Hub...")
    try:
        # Use trust_remote_code=True if the dataset requires custom loading logic.
        ds = load_dataset("callanwu/WebWalkerQA", split="main", trust_remote_code=True)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    train_candidates = []
    test_candidates = []

    print("Processing and filtering the data...")
    # 3. Iterate through and process each item in the dataset.
    for item in ds:
        # You may want to apply your own filtering logic here.
        # For example, we filtered out examples that can not be answered by gpt4.1

        info = item.get("info", {})
        difficulty = info.get('difficulty_level', '')
        lang = info.get('lang', '')

        # Construct the 'problem' field.
        problem = (
            item.get("original_question", "")
            + " You should navigate the website to find the answer. "
            + "The root url is "
            + item.get("root_url", "")
            + ". The answer should be based on the information on the website."
        )
        answer = item.get("expected_answer", "")
        
        simple_item = {"problem": problem, "answer": answer}

        # Split the data into test and train candidates.
        if difficulty == "easy" and lang == "en":
            test_candidates.append(simple_item)
        else:
            train_candidates.append(simple_item)
    
    print(f"Processing complete. Found {len(train_candidates)} training candidates and {len(test_candidates)} test candidates.")

    # 4. Randomly sample from the candidate lists.
    # Or you can filter based on other criteria.
    random.seed(42)
    final_test_list = random.sample(test_candidates, min(test_sample_size, len(test_candidates)))
    final_train_list = random.sample(train_candidates, min(train_sample_size, len(train_candidates)))
    
    print(f"Sampling complete. Final train set size: {len(final_train_list)}, Final test set size: {len(final_test_list)}")

    # 5. Save the data to .jsonl files.
    dataset_splits = {"train": final_train_list, "test": final_test_list}
    
    for split_name, data_list in dataset_splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for record in data_list:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Successfully wrote to {output_file}")

    # 6. Create and save the dataset_dict.json file.
    dataset_info = {
        "citation": "",
        "description": "A custom dataset created from callanwu/WebWalkerQA for RL training.",
        "splits": {
            "train": {"name": "train", "num_examples": len(final_train_list)},
            "test": {"name": "test", "num_examples": len(final_test_list)}
        }
    }
    
    dict_path = os.path.join(output_dir, "dataset_dict.json")
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"Successfully wrote to {dict_path}")
    
    print("\nAll files created successfully!")


if __name__ == "__main__":
    # Define the output directory. You can change this to any path you prefer.
    # This example uses a folder named "webwalker_rl_dataset" in the same directory as the script.
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(current_file_dir, "webwalker_rl_dataset")
    
    # Alternatively, you can use an absolute path like in your original script:
    # output_directory = "/mnt/data/zhangwenhao.zwh/webwalker_rl_dataset"

    # Call the main function to create the dataset files.
    create_dataset_files(
        output_dir=output_directory,
        train_sample_size=160,
        test_sample_size=16
    )

    # --- Verification Step: Test loading the generated dataset ---
    print("\n--- Verifying the created dataset ---")
    try:
        load_ds = load_dataset(output_directory)
        print("Dataset loaded successfully for verification!")
        print(f"Train set size: {len(load_ds['train'])}")
        print(f"Test set size: {len(load_ds['test'])}")
    except Exception as e:
        print(f"Failed to load the created dataset: {e}")

