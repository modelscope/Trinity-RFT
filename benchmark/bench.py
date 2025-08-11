import argparse
import os
import subprocess
import time

import yaml


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
        if args.node_num == 1:  # single node
            for trainer_gpu_num in range(1, args.gpu_per_node):
                if batch_size % trainer_gpu_num == 0:
                    explorer_gpu_num = args.gpu_per_node - trainer_gpu_num
                    if explorer_gpu_num % tensor_parallel_size != 0:
                        continue
                    explorer_num = explorer_gpu_num // tensor_parallel_size
                    ratio = explorer_num / trainer_gpu_num
                    if opt_ratio_diff > abs(ratio - args.explorer_trainer_ratio):
                        opt_ratio_diff = abs(ratio - args.explorer_trainer_ratio)
                        opt_explorer_num = explorer_num
        else:  # multi node
            assert (
                args.gpu_per_node % tensor_parallel_size == 0
            ), f"Please adjust the value of `tensor_parallel_size` so that it is a divisor of `gpu_per_node`."
            for trainer_node_num in range(1, args.node_num):
                trainer_gpu_num = args.gpu_per_node * trainer_node_num
                if batch_size % trainer_gpu_num == 0:
                    explorer_gpu_num = (args.node_num - trainer_node_num) * args.gpu_per_node
                    explorer_num = explorer_gpu_num // tensor_parallel_size
                    ratio = explorer_num / trainer_gpu_num
                    if opt_ratio_diff > abs(ratio - args.explorer_trainer_ratio):
                        opt_ratio_diff = abs(ratio - args.explorer_trainer_ratio)
                        opt_explorer_num = explorer_num
        assert (
            opt_explorer_num is not None
        ), f"Cannot find a suitable explorer number. Please check the value of `train_batch_size`."
        config["explorer"]["rollout_model"]["engine_num"] = opt_explorer_num


def prepare_configs(args):
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_path, "config", f"{args.dataset}-template.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    current_time = time.time()
    current_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(current_time))
    run_path = os.path.join(base_path, "runs", current_time_str)
    os.makedirs(run_path)

    config["name"] += f"-{current_time_str}"
    config["checkpoint_root_dir"] = os.path.join(run_path, "checkpoints")
    set_engine_num(config, args)
    if args.model_path:
        model_path = args.model_path
    elif os.environ.get("MODEL_PATH", None):
        model_path = os.environ.get("MODEL_PATH")
    else:
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    config["model"]["model_path"] = model_path
    if args.lr:
        config["trainer"]["trainer_config"]["actor_rollout_ref"]["actor"]["optim"]["lr"] = args.lr
    if args.sync_interval:
        config["synchronizer"]["sync_interval"] = args.sync_interval

    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return config_path


def main(args):
    config_path = prepare_configs(args)
    cmd_list = [
        "python",
        "-m",
        "trinity.cli.launcher",
        "run",
        "--config",
        config_path,
    ]
    if args.dlc:  # TODO: bug fix
        cmd_list.append("--dlc")
    subprocess.run(cmd_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["gsm8k", "countdown"])
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
    parser.add_argument("--lr", type=float, default=None, help="Specify the learning rate.")
    parser.add_argument(
        "--sync_interval", type=int, default=None, help="Specify the sync interval."
    )
    args = parser.parse_args()
    main(args)
