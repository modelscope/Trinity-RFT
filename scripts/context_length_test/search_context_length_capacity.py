"""Automated context length testing for large language models using distributed training configurations.

This script runs scalability tests on a given model by launching training jobs with increasing
context lengths until OOM (Out-of-Memory) errors occur. It supports sequence parallelism and multiple
GPU configurations.
"""

import argparse
import os
import shutil
import subprocess
import threading
from typing import List, Optional

import transformers
import yaml

from trinity.utils.dlc_utils import is_running, setup_ray_cluster, stop_ray_cluster

# Default list of GPU counts to test
DEFAULT_GPU_NUMS: List[int] = [1, 2, 4, 6]
EXCEPTION_STRING = "Traceback (most recent call last)"
OOM_STRING = "torch.OutOfMemoryError: CUDA out of memory"
CUDA_ERROR_STRING = "RuntimeError: CUDA error:"


def monitor_output(
    pipe,
    exception_event: threading.Event,
    oom_event: threading.Event,
    log_file,
):
    """Monitors the output stream from a subprocess and sets events if target strings are found.

    Reads lines from the provided pipe (e.g., stdout), writes them to the log file, and checks
    whether the output contains the stop or OOM trigger strings. If found, it sets the corresponding
    threading event to signal termination.

    Args:
        pipe: Readable file-like object (e.g., subprocess.stdout).
        exception_event: Threading event set when an exception is detected.
        oom_event: Threading event set when 'torch.OutOfMemoryError: CUDA out of memory' is detected.
        log_file: Open file handle where output is logged.
    """
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            # Write to log and flush immediately
            log_file.write(line)
            log_file.flush()

            # Check for exception
            if EXCEPTION_STRING in line:
                exception_event.set()

            if exception_event.is_set():
                print(line, end="", flush=True)

            # Check for oom
            if OOM_STRING in line or CUDA_ERROR_STRING in line:
                exception_event.set()
                oom_event.set()
                break
    except Exception as e:
        print(f"Error in monitoring thread: {e}")


def run_command_with_monitor(
    command: List[str],
    envs: dict[str, str],
    log_path: str,
    checkpoint_path: str,
    timeout: Optional[int] = None,
    max_retry: int = 10,
) -> bool:
    """Runs a shell command with real-time output monitoring and early termination support.

    Executes the specified command, merges stdout and stderr, logs output to a file, and monitors
    for exception string. If the string appears or a timeout occurs, the process is terminated.

    Retries execution until no other exception event is raised (i.e., until success or OOM).

    Args:
        command: Command to execute, as a list of strings.
        envs: Environment variables to set for the command.
        log_path: Path to the log file where output will be saved.
        checkpoint_path: Path to the checkpoint directory.
        timeout: Optional timeout in seconds before forcing termination.
        max_retry: Maximum number of retries in case of OOM.

    Returns:
        True if the command completed successfully without OOM error; False otherwise.
    """
    retry_flag = True
    success_flag = False
    envs["TRINITY_CHECKPOINT_ROOT_DIR"] = checkpoint_path
    process_env = os.environ.copy()
    process_env.update(envs)

    for _ in range(max_retry):
        # Clean up checkpoint directory before each run
        shutil.rmtree(checkpoint_path, ignore_errors=True)

        exception_event = threading.Event()
        oom_event = threading.Event()
        is_timeout = False

        with open(log_path, "w", encoding="utf-8") as log_file:
            # Start subprocess with merged stdout/stderr
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=process_env,
            )

            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=monitor_output,
                args=(
                    process.stdout,
                    exception_event,
                    oom_event,
                    log_file,
                ),
                daemon=True,
            )
            monitor_thread.start()

            try:
                # Wait for monitor thread or timeout
                if timeout:
                    monitor_thread.join(timeout)
                    if monitor_thread.is_alive():
                        is_timeout = True
                        timeout *= 1.3
                else:
                    monitor_thread.join()

                # Handle process termination based on events
                if exception_event.is_set() or is_timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()

                    if oom_event.is_set():  # CUDA OOM
                        retry_flag = False
                    elif is_timeout:
                        print("Timeout reached, retrying...")
                    else:
                        print("Exception detected, retrying...")

                    success_flag = False
                else:  # no exception, runs successfully
                    retry_flag = False
                    success_flag = True

                # Ensure process has fully terminated
                if process.poll() is None:
                    process.wait()

            except KeyboardInterrupt:
                process.terminate()
                process.wait()

        if not retry_flag:
            break

    return success_flag


def find_max_model_len(
    model_path: str,
    model_config,
    checkpoint_path: str,
    trainer_gpu_num: int,
    sp_num: int,
    base_log_dir: str,
    start_length: int = 4096,
    save_hf_checkpoint: str = "last",
    entropy_saving: bool = False,
    offload: bool = False,
    use_fused_kernels: bool = False,
    trainer_strategy: str = "fsdp",
    timeout: int = 2400,
) -> int:
    """Finds the maximum context length the model can handle under current hardware configuration.

    Iteratively increases the `MAX_MODEL_LEN` value and runs training jobs until an OOM error occurs.
    Uses different YAML config files depending on whether the length exceeds the original max.

    Args:
        model_path: Path to the pretrained model.
        model_config: Loaded Hugging Face model configuration.
        checkpoint_path: Path to the checkpoint directory.
        trainer_gpu_num: Number of GPUs allocated.
        sp_num: Number of sequence parallel groups.
        base_log_dir: Base directory for saving logs.
        start_length: Initial context length to test.
        save_hf_checkpoint: Checkpoint saving strategy.
        entropy_saving: Whether to enable entropy-saving options.
        offload: Whether to offload parameters to CPU.
        use_fused_kernels: Whether to use fused kernels.
        trainer_strategy: Trainer strategy. Only support "fsdp" and "fsdp2" for now.
        timeout: Timeout in seconds for each training job.

    Returns:
        Maximum supported context length before OOM; 0 if search failed.
    """
    checked_length = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(script_dir, "context_length.yaml")
    plugin_dir = os.path.join(script_dir, "workflow")

    length = start_length
    origin_max_len = model_config.max_position_embeddings
    small_step = 4096
    big_step = origin_max_len // 4
    model_name = os.path.basename(model_path)

    while True:
        log_dir = os.path.join(base_log_dir, model_name, f"gpu-{trainer_gpu_num}", f"sp-{sp_num}")
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, f"model_len-{length}.log")
        if trainer_gpu_num >= 8:
            explorer_gpu_num = 8
        else:
            explorer_gpu_num = 1
        total_gpu_num = trainer_gpu_num + explorer_gpu_num

        # Build command
        cmd_env = {
            "GPU_NUM": f"{total_gpu_num}",
            "ENGINE_NUM": f"{explorer_gpu_num}",
            "SP_NUM": f"{sp_num}",
            "REPEAT_TIMES": f"{trainer_gpu_num // sp_num * 8}",
            "MODEL_PATH": f"{model_path}",
            "MAX_MODEL_LEN": f"{length}",
        }
        if length > origin_max_len:
            rope_config = {
                "rope_type": "yarn",
                "factor": length / origin_max_len,
                "original_max_position_embeddings": origin_max_len,
            }
            cmd_env["ROPE_SCALING"] = yaml.dump(rope_config, default_flow_style=True).strip()
        if save_hf_checkpoint != "last":
            cmd_env["SAVE_HF_CHECKPOINT"] = f"{save_hf_checkpoint}"
        if entropy_saving:
            cmd_env["ENTROPY_SAVING"] = "true"
        if offload:
            cmd_env["OFFLOAD"] = "true"
        if use_fused_kernels:
            cmd_env["FUSED_KERNELS"] = "true"
        if trainer_strategy != "fsdp":
            cmd_env["TRAINER_STRATEGY"] = f"{trainer_strategy}"

        cmd_base = [
            "trinity",
            "run",
            "--config",
            yaml_file,
            "--plugin-dir",
            plugin_dir,
        ]

        print(f"Running: {' '.join(f'{k}={v}' for k, v in cmd_env.items())} {' '.join(cmd_base)}")

        # Run with monitoring
        success = run_command_with_monitor(
            cmd_base,
            cmd_env,
            logfile,
            checkpoint_path,
            timeout=timeout,
        )

        if not success:
            break

        checked_length = length

        # Increase step size after exceeding original limit
        if length < origin_max_len:
            length += small_step
        else:
            length += big_step

    if checked_length == 0:
        print(
            f"Search failed for model {model_name} with {trainer_gpu_num} GPUs. "
            "Please check the log file for details."
        )

    return checked_length


def main(args):
    """Main entry point: orchestrates multi-GPU, multi-SP context length testing."""
    if args.dlc:
        cluster_namespace = "search_context_length_capacity"
        setup_ray_cluster(namespace=cluster_namespace)

    if not is_running():
        raise RuntimeError("Ray is not running, please start it by `ray start --head`.")

    os.makedirs(args.log_dir, exist_ok=True)

    model_name = os.path.basename(args.model_path)
    model_config = transformers.AutoConfig.from_pretrained(args.model_path)

    # Map SP group count to starting context length
    sp_num_to_start_length = {sp_num: args.start_length for sp_num in args.test_sp_num}

    for trainer_gpu_num in args.test_gpu_num:
        # Filter valid SP numbers: divides GPU count and attention heads
        sp_list = [
            sp_num
            for sp_num in args.test_sp_num
            if (trainer_gpu_num % sp_num == 0 and model_config.num_attention_heads % sp_num == 0)
        ]

        last_length = 0
        for sp_num in sp_list:
            start_length = max(last_length, sp_num_to_start_length[sp_num])
            max_length = find_max_model_len(
                model_path=args.model_path,
                model_config=model_config,
                checkpoint_path=args.checkpoint_path,
                trainer_gpu_num=trainer_gpu_num,
                sp_num=sp_num,
                base_log_dir=args.log_dir,
                start_length=start_length,
                save_hf_checkpoint=args.save_hf_checkpoint,
                entropy_saving=args.entropy_saving,
                offload=args.offload,
                use_fused_kernels=args.use_fused_kernels,
                trainer_strategy=args.trainer_strategy,
                timeout=args.timeout,
            )
            last_length = max(max_length, args.start_length)
            sp_num_to_start_length[sp_num] = last_length
            print(
                f"model_name = {model_name}, "
                f"trainer_gpu_num = {trainer_gpu_num}, "
                f"sp_num = {sp_num}, "
                f"max_model_len = {max_length}"
            )

    if args.dlc:
        stop_ray_cluster(namespace=cluster_namespace)


if __name__ == "__main__":
    default_log_dir = os.path.join(os.path.dirname(__file__), "logs")
    parser = argparse.ArgumentParser(
        description="Automated context length scalability testing for LLMs."
    )
    parser.add_argument(
        "--start_length",
        type=int,
        default=4096,
        help="Starting context length for testing.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory containing the pretrained models.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=default_log_dir,
        help="Directory to store experiment logs.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=os.environ.get("TRINITY_CHECKPOINT_ROOT_DIR", "./checkpoints/length-test"),
        help="Checkpoint path for testing. "
        "Note that this directory will be deleted during the test, "
        "please specify a path that is not used by other processes.",
    )
    parser.add_argument(
        "--test_gpu_num",
        type=int,
        nargs="*",
        default=DEFAULT_GPU_NUMS,
        help="List of GPU counts to test.",
    )
    parser.add_argument(
        "--test_sp_num",
        type=int,
        nargs="*",
        default=[1],
        help="List of sequence parallel sizes to test.",
    )
    parser.add_argument(
        "--save_hf_checkpoint",
        type=str,
        choices=["always", "never", "last"],
        default="last",
        help="Whether to save HF checkpoint.",
    )
    parser.add_argument(
        "--entropy_saving",
        action="store_true",
        help="Whether to reduce entropy memory usage.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload model to CPU.",
    )
    parser.add_argument(
        "--use_fused_kernels",
        action="store_true",
        help="Whether to use fused kernels.",
    )
    parser.add_argument(
        "--trainer_strategy",
        type=str,
        choices=["fsdp", "fsdp2"],
        default="fsdp",
        help="Trainer strategy to use.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=2400,
        help="Base timeout duration per experiment in seconds. "
        "Each retry increases the timeout by 30% (multiplied by 1.3).",
    )
    parser.add_argument(
        "--dlc", action="store_true", help="Specify when running in Aliyun PAI DLC."
    )

    args = parser.parse_args()
    main(args)
