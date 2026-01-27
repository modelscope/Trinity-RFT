import argparse
import atexit
import json
import os
import subprocess
import time

# record all servers
servers = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_manager",
        type=str,
        default="conda",
        help="The environment manager program. It's conda in default.",
    )
    parser.add_argument("--log_dir", type=str, default="logs/", help="The directory to store logs.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    env_mapping_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "environments", "env_mapping.json"
    )
    with open(env_mapping_file, "r") as f:
        env_mapping = json.load(f)
    for env_path, env_config in env_mapping.items():
        env_name = env_config["env_name"]
        print(f"Starting server for module [{env_name}]...")
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        with open(os.path.join(args.log_dir, f"{env_name}_{timestamp}_log.txt"), "w") as log_file:
            server = subprocess.Popen(
                f'{args.env_manager} run --no-capture-output -n {env_name} python {env_config["env_entry"]}',
                stdout=log_file,
                stderr=log_file,
                shell=True,
            )
            servers.append(server)
            print(f"Server of module [{env_name}] is started with PID {server.pid}")
    for server in servers:
        server.wait()


def cleanup():
    for server in servers:
        server.terminate()
        server.wait()
        print(f"Server with PID {server.pid} is terminated")
    servers.clear()


# register to clean up all servers when the script exits or terminates
atexit.register(cleanup)

if __name__ == "__main__":
    main()
