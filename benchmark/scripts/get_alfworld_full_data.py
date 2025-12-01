import argparse
import os
import subprocess
import sys

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "alfworld"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    # Step 1: Get all game files from Huggingface
    game_data_dir = os.path.join(args.local_dir, "..", "alfworld_game_data")
    if os.path.exists(game_data_dir) and os.path.exists(os.path.join(game_data_dir, "json_2.1.1")):
        print(f"Game data directory already exists: {game_data_dir}")

    else:
        os.makedirs(game_data_dir, exist_ok=True)
        subprocess.run(["pip", "install", "alfworld[full]"], check=True)
        # Set environment variable for alfworld-download command
        env = os.environ.copy()
        env["ALFWORLD_DATA"] = game_data_dir
        subprocess.run(["alfworld-download"], check=True, env=env)

    # Step 2: Run the script to get the mapping file
    base_dir = os.path.dirname(__file__)
    data_prepare_path = os.path.abspath(
        os.path.join(
            base_dir,
            "..",
            "..",
            "examples",
            "grpo_alfworld",
            "get_alfworld_data.py",
        )
    )
    subprocess.executable(
        [
            sys.executable,
            data_prepare_path,
            "--game_data_path",
            game_data_dir,
            "--local_dir",
            args.local_dir,
        ],
        check=True,
    )
