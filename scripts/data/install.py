import argparse
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_manager",
        type=str,
        default="conda",
        help="The environment manager program. It's conda in default.",
    )
    args = parser.parse_args()

    env_mng = args.env_manager
    print(f"Using environment manager of [{env_mng}].")

    env_mapping_file = os.path.join(
        os.path.dirname(__file__), "..", "environments", "env_mapping.json"
    )
    with open(env_mapping_file, "r") as f:
        env_mapping = json.load(f)
    for env_path, env_config in env_mapping.items():
        env_name = env_config["env_name"]
        print(f"Installing dependencies for module [{env_name}]...")
        # check if it's existing
        res = subprocess.run(
            f"{env_mng} env list | grep {env_name}", shell=True, text=True, stdout=subprocess.PIPE
        )
        if res.returncode == 0 and env_name in res.stdout:
            print(f"Environment [{env_name}] already exists. Skipping...")
        else:
            res = subprocess.run(
                f'{env_mng} env create -f {env_config["env_yaml"]}'
                f"&& {env_mng} init"
                f'&& {env_mng} run -n {env_name} pip install -e ".[dev]"',
                shell=True,
            )
            if res.returncode == 0:
                print(f"Environment [{env_name}] created successfully.")
            else:
                print(f"Failed to create environment [{env_name}] with exit code {res.returncode}.")


if __name__ == "__main__":
    main()
