import argparse
import os
import subprocess
import time

import torch
import torch.distributed as dist
import yaml

from trinity.algorithm.algorithm import ALGORITHM_TYPE
from trinity.common.constants import MODEL_PATH_ENV_VAR
from trinity.utils.dlc_utils import get_dlc_env_vars


def set_engine_num(config, args):
    config["cluster"]["node_num"] = args.node_num
    config["cluster"]["gpu_per_node"] = args.gpu_per_node
    batch_size = config["buffer"]["batch_size"]
    if config["mode"] == "train":
        return

    if args.vllm_tp_size is not None:
        config["explorer"]["rollout_model"]["tensor_parallel_size"] = args.vllm_tp_size
    tensor_parallel_size = config["explorer"]["rollout_model"]["tensor_parallel_size"]

    if args.vllm_engine_num is not None:
        config["explorer"]["rollout_model"]["engine_num"] = args.vllm_engine_num
    else:  # auto set engine_num
        opt_explorer_num, opt_ratio_diff = None, float("inf")
        total_gpu_num = args.node_num * args.gpu_per_node

        def update_opt_explorer_num(trainer_gpu_num, opt_explorer_num, opt_ratio_diff):
            if batch_size % trainer_gpu_num != 0:
                return opt_explorer_num, opt_ratio_diff
            explorer_gpu_num = total_gpu_num - trainer_gpu_num
            if explorer_gpu_num % tensor_parallel_size != 0:
                return opt_explorer_num, opt_ratio_diff
            explorer_num = explorer_gpu_num // tensor_parallel_size
            ratio = explorer_num / trainer_gpu_num
            if opt_ratio_diff > abs(ratio - args.explorer_trainer_ratio):
                return explorer_num, abs(ratio - args.explorer_trainer_ratio)
            return opt_explorer_num, opt_ratio_diff

        if args.node_num == 1:  # single node
            for trainer_gpu_num in range(1, args.gpu_per_node):
                opt_explorer_num, opt_ratio_diff = update_opt_explorer_num(
                    trainer_gpu_num, opt_explorer_num, opt_ratio_diff
                )
        else:  # multi node
            assert (
                args.gpu_per_node % tensor_parallel_size == 0
            ), "Please adjust the value of `tensor_parallel_size` so that it is a divisor of `gpu_per_node`."
            for trainer_node_num in range(1, args.node_num):
                trainer_gpu_num = args.gpu_per_node * trainer_node_num
                opt_explorer_num, opt_ratio_diff = update_opt_explorer_num(
                    trainer_gpu_num, opt_explorer_num, opt_ratio_diff
                )
        assert (
            opt_explorer_num is not None
        ), "Cannot find a suitable explorer number. Please check the value of `train_batch_size`."
        config["explorer"]["rollout_model"]["engine_num"] = opt_explorer_num


def prepare_configs(args, rank, current_time):
    base_path = os.path.dirname(os.path.abspath(__file__))

    current_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(current_time))
    run_path = os.path.join(base_path, "runs", current_time_str)
    config_path = os.path.join(run_path, "config.yaml")
    if rank == 0:
        os.makedirs(run_path)

        with open(os.path.join(base_path, "config", f"{args.dataset}-template.yaml")) as f:
            config = yaml.safe_load(f)

        config["name"] += f"-{current_time_str}"
        config["checkpoint_root_dir"] = os.path.join(run_path, "checkpoints")
        set_engine_num(config, args)
        config["model"]["model_path"] = (
            args.model_path
            or config["model"]["model_path"]
            or os.environ.get(MODEL_PATH_ENV_VAR, "Qwen/Qwen2.5-1.5B-Instruct")
        )
        if ALGORITHM_TYPE.get(config["algorithm"]["algorithm_type"]).use_critic:
            config["model"]["critic_model_path"] = (
                args.critic_model_path
                or config["model"].get("critic_model_path")
                or config["model"]["model_path"]
            )
            if args.critic_lr:
                config["trainer"]["trainer_config"]["critic"]["optim"]["lr"] = args.critic_lr
        config["buffer"]["explorer_input"]["taskset"]["path"] = (
            args.taskset_path
            or os.environ.get("TASKSET_PATH")
            or config["buffer"]["explorer_input"]["taskset"]["path"]
        )
        assert (
            config["buffer"]["explorer_input"]["taskset"]["path"] is not None
        ), "Please specify taskset path."
        if args.lr:
            config["algorithm"]["optimizer"]["lr"] = args.lr
        if args.sync_interval:
            config["synchronizer"]["sync_interval"] = args.sync_interval

        with open(config_path, "w") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return config_path


def setup_dlc():
    envs = get_dlc_env_vars()
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=envs["WORLD_SIZE"],
        rank=envs["RANK"],
    )
    if envs["RANK"] == 0:
        current_time = time.time()
        time_tensor = torch.tensor([current_time], device="cpu")
    else:
        time_tensor = torch.tensor([0.0], device="cpu")
    dist.broadcast(time_tensor, src=0)
    return envs["RANK"], time_tensor.item()


def main(args):
    if args.dlc:
        rank, current_time = setup_dlc()
    else:
        rank, current_time = 0, time.time()
    config_path = prepare_configs(args, rank, current_time)
    cmd_list = [
        "python",
        "-m",
        "trinity.cli.launcher",
        "run",
        "--config",
        config_path,
    ]
    if args.dlc:
        dist.barrier()
        dist.destroy_process_group()
        cmd_list.append("--dlc")
    subprocess.run(cmd_list, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["gsm8k", "countdown", "openr1"])
    parser.add_argument(
        "--dlc", action="store_true", help="Specify when running in Aliyun PAI DLC."
    )
    parser.add_argument("--node_num", type=int, default=1, help="Specify the number of nodes.")
    parser.add_argument(
        "--gpu_per_node", type=int, default=8, help="Specify the number of GPUs per node."
    )
    parser.add_argument(
        "--vllm_engine_num", type=int, default=None, help="Specify the number of vLLM engines."
    )
    parser.add_argument(
        "--vllm_tp_size", type=int, default=None, help="Specify the number of vLLM tp size."
    )
    parser.add_argument(
        "--explorer_trainer_ratio",
        type=float,
        default=0.6,
        help="Specify the ratio of explorer engine num to trainer gpu num.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Specify the path to the model checkpoint.",
    )
    parser.add_argument(
        "--critic_model_path",
        type=str,
        default=None,
        help="Specify the path to the critic model checkpoint.",
    )
    parser.add_argument(
        "--taskset_path", type=str, default=None, help="Specify the path to the taskset."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Specify the learning rate for actor model."
    )
    parser.add_argument(
        "--critic_lr", type=float, default=None, help="Specify the learning rate for critic model."
    )
    parser.add_argument(
        "--sync_interval", type=int, default=None, help="Specify the sync interval."
    )
    args = parser.parse_args()
    main(args)
