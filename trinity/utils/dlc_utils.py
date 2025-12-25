import os
import subprocess
import sys
import time

import ray

from trinity.utils.log import get_logger

logger = get_logger(__name__)

CLUSTER_ACTOR_NAME = "cluster_status"


class ClusterStatus:
    def __init__(self):
        self.finished = False

    def finish(self) -> None:
        self.finished = True

    def running(self) -> bool:
        return not self.finished


def get_dlc_env_vars() -> dict:
    envs = {
        "RANK": int(os.environ.get("RANK", -1)),  # type: ignore
        "WORLD_SIZE": int(os.environ.get("WORLD_SIZE", -1)),  # type: ignore
        "MASTER_ADDR": os.environ.get("MASTER_ADDR", None),
        "MASTER_PORT": os.environ.get("MASTER_PORT", None),
    }
    for key, value in envs.items():
        if value is None or value == -1:
            logger.error(f"DLC env var `{key}` is not set.")
            raise ValueError(f"DLC env var `{key}` is not set.")
    return envs


def is_running() -> bool:
    """Check if ray cluster is running."""
    python_dir = os.path.dirname(sys.executable)
    ray_path = os.path.join(python_dir, "ray")
    if not os.path.exists(ray_path):
        ray_path = "ray"
    ret = subprocess.run([ray_path, "status"], capture_output=True)
    return ret.returncode == 0


def wait_for_ray_setup() -> None:
    while True:
        if is_running():
            break
        else:
            logger.info("Waiting for ray cluster to be ready...")
            time.sleep(1)


def wait_for_ray_worker_nodes(world_size: int) -> None:
    while True:
        alive_nodes = [node for node in ray.nodes() if node["Alive"]]
        if len(alive_nodes) >= world_size:
            break
        else:
            logger.info(
                f"{len(alive_nodes)} nodes have joined so far, waiting for {world_size - len(alive_nodes)} nodes..."
            )
            time.sleep(1)


def setup_ray_cluster(namespace: str) -> str:
    """Setup a ray cluster in DLC environment.

    This function will start a ray cluster if it is not running, otherwise it will reuse the existing ray cluster.

    Returns:
        str: The address of the ray cluster.
    """
    env_vars = get_dlc_env_vars()
    is_master = env_vars["RANK"] == 0

    if is_running():
        # reuse existing ray cluster
        return "auto"
    else:
        if is_master:
            cmd = f"ray start --head --port={env_vars['MASTER_PORT']} --node-ip-address={env_vars['MASTER_ADDR']}"
        else:
            cmd = f"ray start --address={env_vars['MASTER_ADDR']}:{env_vars['MASTER_PORT']}"
        ret = subprocess.run(cmd, shell=True, capture_output=True)
        logger.info(f"Starting ray cluster: {cmd}")
        if ret.returncode != 0:
            logger.error(f"Failed to start ray cluster: {cmd}")
            logger.error(f"ret.stdout: {ret.stdout!r}")
            logger.error(f"ret.stderr: {ret.stderr!r}")
            sys.exit(1)

        wait_for_ray_setup()
        time.sleep(5)
        ray.init(
            address=f"{env_vars['MASTER_ADDR']}:{env_vars['MASTER_PORT']}",
            namespace=namespace,
            ignore_reinit_error=True,
        )
        if is_master:
            # master wait for worker nodes to join
            wait_for_ray_worker_nodes(env_vars["WORLD_SIZE"])
            ray.shutdown()
            return f"{env_vars['MASTER_ADDR']}:{env_vars['MASTER_PORT']}"
        else:
            # worker wait on the cluster status actor
            cluster_status = (
                ray.remote(ClusterStatus)
                .options(
                    name=CLUSTER_ACTOR_NAME,
                    namespace=namespace,
                    get_if_exists=True,
                )
                .remote()
            )
            while True:
                if ray.get(cluster_status.running.remote()):
                    ret = subprocess.run("ray status", shell=True, capture_output=True)
                    print(ret.stdout.decode())
                    time.sleep(5)
                else:
                    logger.info("Ray cluster is not running, exiting.")
                    break
            sys.exit(0)


def stop_ray_cluster(namespace: str):
    """
    Stop the ray cluster by sending a signal to the cluster status actor.
    """
    cluster_status = (
        ray.remote(ClusterStatus)
        .options(
            name=CLUSTER_ACTOR_NAME,
            namespace=namespace,
            get_if_exists=True,
        )
        .remote()
    )
    ray.get(cluster_status.finish.remote())
    logger.info("Stopping ray cluster...")
