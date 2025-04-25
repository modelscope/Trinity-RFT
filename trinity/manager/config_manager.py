import os
from typing import List

import streamlit as st
import yaml

from trinity.common.constants import AlgorithmType
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows.workflow import WORKFLOWS
from trinity.trainer.verl.ray_trainer import AdvantageEstimator


class ConfigManager:
    def __init__(self):
        self._init_default_config()
        self.unfinished_fields = set()
        st.set_page_config(page_title="Trainer Config Generator", page_icon=":robot:")
        st.title("Trainer Config Generator")
        if "_init_config_manager" not in st.session_state:
            self.reset_session_state()
        self.maintain_session_state()
        mode = st.pills(
            "Select Mode",
            options=["Beginner Mode", "Expert Mode"],
            default="Beginner Mode",
            label_visibility="collapsed",
        )
        if mode == "Beginner Mode":
            self.beginner_mode()
        else:
            self.expert_mode()
        self.generate_config()

    def _init_default_config(self):
        self.default_config = {
            "_init_config_manager": True,
            "project": "Trinity-RFT",
            "exp_name": "qwen2.5-1.5B",
            # Model Configs
            "model_path": "",
            "critic_model_path": "",
            "checkpoint_path": "",
            "node_num": 1,
            "gpu_per_node": 8,
            "total_gpu_num": 8,
            "trainer_gpu_num": 6,
            "max_prompt_tokens": 1024,
            "max_response_tokens": 1024,
            # Data and Buffer Configs
            "total_epoch": 20,
            "task_num_per_batch": 6,
            "dataset_path": "",
            "train_split": "train",
            "eval_split": "",
            "prompt_key": "question",
            "response_key": "answer",
            "default_workflow_type": "math_workflow",
            "default_reward_fn_type": "math_reward",
            "storage_type": "queue",
            "db_url": "",
            "max_retry_times": 3,
            "max_retry_interval": 1,
            "sft_warmup_dataset_path": "",
            "sft_warmup_train_split": "train",
            "sft_warmup_eval_split": "",
            "sft_warmup_prompt_key": "question",
            "sft_warmup_response_key": "answer",
            # Explorer and Sync Configs
            "engine_type": "vllm_async",
            "engine_num": 2,
            "tensor_parallel_size": 1,
            "repeat_times": 1,
            "sync_method": "online",
            "sync_iteration_interval": 10,
            "runner_num": 32,
            "max_pending_requests": 32,
            "max_waiting_steps": 4,
            "dtype": "bfloat16",
            "backend": "nccl",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 42,
            "logprobs": None,
            "enable_prefix_caching": False,
            "enforce_eager": True,
            # Trainer Configs
            "trainer_type": "verl",
            "algorithm_type": AlgorithmType.PPO.value,
            "sft_warmup_iteration": 0,
            "eval_interval": 1000,
            "trainer_config_path": "",
            # veRL Trainer Configs
            "training_args": [
                "balance_batch",
                "gradient_checkpointing",
                "remove_padding",
                "dynamic_bsz",
            ],
            "save_freq": 100,
            "training_strategy": "fsdp",
            "param_offload": False,
            "optimizer_offload": False,
            "resume_mode": "auto",
            "resume_from_path": "",
            "critic_warmup": 0,
            "total_training_steps": None,
            "default_hdfs_dir": None,
            "remove_previous_ckpt_in_save": False,
            "del_local_ckpt_after_load": False,
            "max_actor_ckpt_to_keep": None,
            "max_critic_ckpt_to_keep": None,
            "gamma": 1.0,
            "lam": 1.0,
            "adv_estimator": "gae",
            "kl_penalty": "low_var_kl",
            "kl_ctrl_type": "fixed",
            "kl_ctrl_coef": 0.001,
            "actor_ppo_micro_batch_size_per_gpu": 4,
            "ref_log_prob_micro_batch_size_per_gpu": 8,
            "actor_ulysses_sequence_parallel_size": 1,
            "actor_lr": 1e-6,
            "actor_warmup_style": "constant",
            "actor_lr_warmup_steps_ratio": 0.0,
            "actor_tau": 0.0,
            "actor_opmd_baseline": "mean",
            "actor_use_uid": False,
            "actor_grad_clip": 1.0,
            "actor_clip_ratio": 0.2,
            "actor_entropy_coeff": 0.001,
            "actor_use_kl_loss": False,
            "actor_kl_loss_coef": 0.001,
            "actor_kl_loss_type": "low_var_kl",
            "actor_checkpoint": ["model", "hf_model", "optimizer", "extra"],
            "critic_lr": 1e-6,
            "critic_warmup_style": "constant",
            "critic_lr_warmup_steps_ratio": 0.0,
            "critic_grad_clip": 1.0,
            "critic_cliprange_value": 0.5,
            "critic_ppo_micro_batch_size_per_gpu": 8,
            "critic_ulysses_sequence_parallel_size": 1,
            "training_mode": "PPO",
        }

    def reset_session_state(self):
        for key, value in self.default_config.items():
            st.session_state[key] = value

    def maintain_session_state(self):
        for key in self.default_config:
            st.session_state[key] = st.session_state[key]

    def _set_project(self):
        st.text_input("Project", key="project")

    def _set_name(self):
        st.text_input("Experiment Name", key="exp_name")

    def _set_model_path(self):
        st.text_input("Model Path", key="model_path")
        if not st.session_state["model_path"].strip():
            self.unfinished_fields.add("model_path")
            st.warning("Please input model path.")
        elif not os.path.isabs(st.session_state["model_path"].strip()):
            self.unfinished_fields.add("model_path")
            st.warning("Please input an absolute path.")

    def _set_critic_model_path(self):
        st.text_input(
            "Critic Model Path (defaults to `model_path`)",
            key="critic_model_path",
        )

    def _set_checkpoint_path(self):
        st.text_input("Checkpoint Path", key="checkpoint_path")
        if not st.session_state["checkpoint_path"].strip():  # TODO: may auto generate
            self.unfinished_fields.add("checkpoint_path")
            st.warning("Please input checkpoint path.")
        elif not os.path.isabs(st.session_state["checkpoint_path"].strip()):
            self.unfinished_fields.add("checkpoint_path")
            st.warning("Please input an absolute path.")

    def _set_node_num(self):
        st.number_input("Node Num", key="node_num", min_value=1, on_change=self._set_total_gpu_num)

    def _set_gpu_per_node(self):
        st.number_input(
            "GPU Per Node",
            key="gpu_per_node",
            min_value=1,
            max_value=8,
            on_change=self._set_total_gpu_num,
        )

    def _set_total_gpu_num(self):
        st.session_state["total_gpu_num"] = (
            st.session_state["gpu_per_node"] * st.session_state["node_num"]
        )
        self._set_trainer_gpu_num()

    def _set_trainer_gpu_num(self):
        st.session_state["trainer_gpu_num"] = (
            st.session_state["total_gpu_num"]
            - st.session_state["engine_num"] * st.session_state["tensor_parallel_size"]
        )
        print(f'{st.session_state["trainer_gpu_num"]=}')

    def _set_max_prompt_tokens(self):
        st.number_input("Max Prompt Tokens", key="max_prompt_tokens", min_value=1)

    def _set_max_response_tokens(self):
        st.number_input("Max Response Tokens", key="max_response_tokens", min_value=1)

    def _set_total_epoch(self):
        st.number_input("Total Epoch", key="total_epoch", min_value=1)

    @property
    def _str_for_task_num_per_batch(self):
        return f"`task_num_per_batch` must be an multiple of `gpu_per_node * node_num - engine_num * tensor_parallel_size` = {st.session_state['trainer_gpu_num']}"

    def _set_task_num_per_batch(self):
        trainer_gpu_num = st.session_state["trainer_gpu_num"]
        if st.session_state["task_num_per_batch"] < trainer_gpu_num:
            st.session_state["task_num_per_batch"] = trainer_gpu_num
        st.number_input(
            "Task Num Per Batch",
            key="task_num_per_batch",
            min_value=trainer_gpu_num,
            step=trainer_gpu_num,
            help=self._str_for_task_num_per_batch,
        )

    def _check_task_num_per_batch(self):
        if st.session_state["task_num_per_batch"] % st.session_state["trainer_gpu_num"] != 0:
            self.unfinished_fields.add("task_num_per_batch")
            st.warning(self._str_for_task_num_per_batch)

    def _set_dataset_path(self):
        st.text_input("Dataset Path", key="dataset_path")
        if not st.session_state["dataset_path"].strip():
            self.unfinished_fields.add("dataset_path")
            st.warning("Please input dataset path.")
        elif not os.path.isabs(st.session_state["dataset_path"].strip()):
            self.unfinished_fields.add("dataset_path")
            st.warning("Please input an absolute path.")

    def _set_dataset_args(self):
        if st.session_state["dataset_path"] and "://" not in st.session_state["dataset_path"]:
            train_split_col, eval_split_col, prompt_key_col, response_key_col = st.columns(4)
            train_split_col.text_input("Train Split", key="train_split")
            eval_split_col.text_input("Eval Split", key="eval_split")
            prompt_key_col.text_input("Prompt Key", key="prompt_key")
            response_key_col.text_input("Response Key", key="response_key")

    def _set_default_workflow_type(self):  # TODO: add help messages
        st.selectbox(
            "Default Workflow Type",
            WORKFLOWS.modules.keys(),
            key="default_workflow_type",
            help=r"""`simple_workflow`: call 'model.chat()' to get responses.

`math_workflow`: call 'model.chat()' with a pre-defined system prompt to get responses.

Other workflows: conduct multi-turn task for the given dataset.
""",
        )

    def _set_default_reward_fn_type(self):  # TODO: add help messages
        st.selectbox(
            "Default Reward Fn Type",
            REWARD_FUNCTIONS.modules.keys(),
            key="default_reward_fn_type",
            help=r"""`accuracy_reward`: check the accuracy for math problems.

`format_reward`: check if the response matches the format (default: `<think>*</think>* <answer>*</answer>`).

`math_reward`: `accuracy_reward` (1 or 0) + `format_reward` (+0.1 or -0.1).
""",
        )

    def _set_storage_type(self):
        st.selectbox(
            "Storage Type",
            ["sql", "redis", "queue"],
            key="storage_type",
        )

    def _set_db_url(self):
        st.text_input(
            "DB URL",
            key="db_url",
            help=r"Default to `sqlite:///{os.path.join(checkpoint_path, '.cache', project_name, experiment_name)}/data.db`",
        )

    def _set_max_retry_times(self):
        st.number_input("Max Retry Times", key="max_retry_times", min_value=1)

    def _set_max_retry_interval(self):
        st.number_input("Max Retry Interval", key="max_retry_interval", min_value=1)

    def _check_sft_warmup_dataset_path(self):
        if st.session_state["sft_warmup_iteration"]:
            if not st.session_state["sft_warmup_dataset_path"].strip():
                self.unfinished_fields.add("sft_warmup_dataset_path")
                st.warning(
                    "Please input SFT warmup dataset path when `sft_warmup_iteration` is not 0"
                )
            elif not os.path.isabs(st.session_state["sft_warmup_dataset_path"].strip()):
                self.unfinished_fields.add("sft_warmup_dataset_path")
                st.warning("Please input an absolute path.")

    def _set_sft_warmup_dataset_path(self):
        st.text_input("SFT Warmup Dataset Path", key="sft_warmup_dataset_path")
        self._check_sft_warmup_dataset_path()

    def _set_sft_warmup_dataset_args(self):
        if (
            st.session_state["sft_warmup_dataset_path"]
            and "://" not in st.session_state["sft_warmup_dataset_path"]
        ):  # TODO
            (
                sft_warmup_train_split_col,
                sft_warmup_eval_split_col,
                sft_warmup_prompt_key_col,
                sft_warmup_response_key_col,
            ) = st.columns(4)
            sft_warmup_train_split_col.text_input("SFT Train Split", key="sft_warmup_train_split")
            sft_warmup_eval_split_col.text_input("SFT Eval Split", key="sft_warmup_eval_split")
            sft_warmup_prompt_key_col.text_input("SFT Prompt Key", key="sft_warmup_prompt_key")
            sft_warmup_response_key_col.text_input(
                "SFT Response Key", key="sft_warmup_response_key"
            )

    def _set_engine_type(self):
        st.selectbox("Explorer Engine Type", ["vllm_async", "vllm"], key="engine_type")

    @property
    def _str_for_engine_num_and_tp_size(self):
        return r"""```python
assert engine_num * tensor_parallel_size < gpu_per_node * node_num
if node_num > 1:
    assert gpu_per_node % tensor_parallel_size == 0
    assert engine_num * tensor_parallel_size % gpu_per_node == 0
```"""

    def _set_engine_num(self):
        total_gpu_num = st.session_state["gpu_per_node"] * st.session_state["node_num"]
        max_engine_num = (total_gpu_num - 1) // st.session_state["tensor_parallel_size"]
        if st.session_state["engine_num"] > max_engine_num:
            st.session_state["engine_num"] = max_engine_num
            self._set_trainer_gpu_num()
        st.number_input(
            "Inference Engine Num",
            key="engine_num",
            min_value=1,
            max_value=max_engine_num,
            help=f"`engine_num` must meet the following constraints:\n{self._str_for_engine_num_and_tp_size}",
            on_change=self._set_trainer_gpu_num,
        )

    def _set_tensor_parallel_size(self):
        total_gpu_num = st.session_state["gpu_per_node"] * st.session_state["node_num"]
        max_tensor_parallel_size = (total_gpu_num - 1) // st.session_state["engine_num"]
        if st.session_state["tensor_parallel_size"] > max_tensor_parallel_size:
            st.session_state["tensor_parallel_size"] = max_tensor_parallel_size
            self._set_trainer_gpu_num()
        st.number_input(
            "Tensor Parallel Size",
            key="tensor_parallel_size",
            min_value=1,
            max_value=max_tensor_parallel_size,
            help=f"`tensor_parallel_size` must meet the following constraints:\n{self._str_for_engine_num_and_tp_size}",
            on_change=self._set_trainer_gpu_num,
        )

    def _check_engine_num_and_tp_size(self):
        node_num = st.session_state["node_num"]
        gpu_per_node = st.session_state["gpu_per_node"]
        engine_num = st.session_state["engine_num"]
        tensor_parallel_size = st.session_state["tensor_parallel_size"]
        if node_num > 1:
            if gpu_per_node % tensor_parallel_size != 0:
                self.unfinished_fields.add("tensor_parallel_size")
                st.warning(
                    "Please ensure that `tensor_parallel_size` is a factor of `gpu_per_node` when `node_num > 1`."
                )
            if engine_num * tensor_parallel_size % gpu_per_node != 0:
                self.unfinished_fields.add("engine_num")
                st.warning(
                    "Please ensure that `engine_num * tensor_parallel_size` can be divided by `gpu_per_node` when `node_num > 1`."
                )

    def _set_repeat_times(self):
        st.number_input("Repeat Times", key="repeat_times", min_value=1)

    def _set_sync_method(self):
        st.selectbox("Sync Method", ["online", "offline"], key="sync_method")

    def _set_sync_iteration_interval(self):
        st.number_input(
            "Sync Iteration Interval",
            key="sync_iteration_interval",
            min_value=1,
            help="""The iteration interval at which the `explorer` and `trainer` synchronize and switch.""",
        )

    def _set_runner_num(self):
        st.number_input("Runner Num", key="runner_num", min_value=1)

    def _set_max_pending_requests(self):
        st.number_input("Max Pending Requests", key="max_pending_requests", min_value=1)

    def _set_max_waiting_steps(self):
        st.number_input("Max Waiting Steps", key="max_waiting_steps", min_value=1)

    def _set_dtype(self):
        st.selectbox("Dtype", ["float16", "bfloat16", "float32"], key="dtype")

    def _set_backend(self):
        st.selectbox("Backend", ["nccl"], key="backend")

    def _set_temperature(self):
        st.number_input("Temperature", key="temperature", min_value=0.0, max_value=2.0)

    def _set_top_p(self):
        st.number_input("Top P", key="top_p", min_value=0.0, max_value=1.0)

    def _set_top_k(self):
        st.number_input(
            "Top K",
            key="top_k",
            min_value=-1,
            max_value=512,
            help="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.",
        )

    def _set_seed(self):
        st.number_input("Seed", key="seed", step=1)

    def _set_logprobs(self):
        st.number_input("Logprobs", key="logprobs", min_value=0, max_value=20)

    def _set_enable_prefix_caching(self):
        st.checkbox("Enable Prefix Caching", key="enable_prefix_caching")

    def _set_enforce_eager(self):
        st.checkbox("Enforce Eager", key="enforce_eager")

    def _set_trainer_type(self):
        st.selectbox("Trainer Type", ["verl"], key="trainer_type")

    def _set_algorithm_type(self):
        st.selectbox(
            "Algorithm Type",
            [AlgorithmType.PPO.value, AlgorithmType.DPO.value, AlgorithmType.OPMD.value],
            key="algorithm_type",
        )

    def _set_sft_warmup_iteration(self):
        st.number_input("SFT Warmup Iteration", key="sft_warmup_iteration", min_value=0)

    def _set_eval_interval(self):
        st.number_input("Eval Interval", key="eval_interval", min_value=1)

    def _set_trainer_config_path(self):
        st.text_input("Trainer Config Path", key="trainer_config_path")
        if not st.session_state["trainer_config_path"].strip():
            self.unfinished_fields.add("trainer_config_path")
            st.warning("Please input trainer config path")
        elif not os.path.isabs(st.session_state["trainer_config_path"].strip()):
            self.unfinished_fields.add("trainer_config_path")
            st.warning("Please input an absolute path.")

    def _set_training_args(self):
        st.multiselect(
            "Training Args",
            [
                "balance_batch",
                "gradient_checkpointing",
                "remove_padding",
                "dynamic_bsz",
            ],
            key="training_args",
        )

    def _set_save_freq(self):
        if st.session_state["sync_method"] == "online":
            freeze_save_freq = False
        else:
            st.session_state["save_freq"] = st.session_state["sync_iteration_interval"]
            freeze_save_freq = True
        st.number_input(
            "Save Freq",
            key="save_freq",
            min_value=1,
            help="Set to `sync_iteration_interval` when `sync_method` is `offline`",
            disabled=freeze_save_freq,
        )

    def _set_training_strategy(self):
        st.selectbox(
            "Training Strategy",
            ["fsdp", "megatron"],
            key="training_strategy",
            help="megatron is not tested",
        )

    def _set_param_offload(self):
        st.checkbox("FSDP Param Offload", key="param_offload")

    def _set_optimizer_offload(self):
        st.checkbox("FSDP Optimizer Offload", key="optimizer_offload")

    def _set_resume_mode(self):
        st.selectbox("Resume Mode", ["disable", "auto", "resume_path"], key="resume_mode")

    def _set_resume_from_path(self):
        if st.session_state["resume_mode"] == "resume_path":
            st.text_input("Resume Path", key="resume_from_path")
            if (
                not st.session_state["resume_from_path"].strip()
                or "global_step_" not in st.session_state["resume_from_path"]
            ):
                self.unfinished_fields.add("resume_from_path")
                st.warning("Please input a valid resume path when `resume_mode == resume_path`")

    def _set_critic_warmup(self):
        st.number_input("Critic Warmup Iteration", key="critic_warmup", min_value=0)

    def _set_total_training_steps(self):
        st.number_input("Total Training Steps", key="total_training_steps", min_value=1)

    def _set_default_hdfs_dir(self):
        st.text_input("Default HDFS Dir", key="default_hdfs_dir")

    def _set_remove_previous_ckpt_in_save(self):
        st.checkbox("Remove Previous Checkpoint in Save", key="remove_previous_ckpt_in_save")

    def _set_del_local_ckpt_after_load(self):
        st.checkbox("Delete Local Checkpoint After Load", key="del_local_ckpt_after_load")

    def _set_max_actor_ckpt_to_keep(self):
        st.number_input("Max Actor Checkpoint to Keep", key="max_actor_ckpt_to_keep", min_value=1)

    def _set_max_critic_ckpt_to_keep(self):
        st.number_input("Max Critic Checkpoint to Keep", key="max_critic_ckpt_to_keep", min_value=1)

    def _set_gamma(self):
        st.number_input("Gamma", key="gamma")

    def _set_lam(self):
        st.number_input("lam", key="lam")

    def _set_adv_estimator(self):
        st.selectbox(
            "Advantage Estimator",
            [member.value for member in AdvantageEstimator],
            key="adv_estimator",
        )

    def _set_kl_penalty(self):
        st.selectbox("KL Penalty", ["kl", "abs", "mse", "low_var_kl"], key="kl_penalty")

    def _set_kl_ctrl_type(self):
        st.selectbox("KL Ctrl Type", ["fixed", "adaptive"], key="kl_ctrl_type")

    def _set_kl_ctrl_coef(self):
        st.number_input("KL Ctrl Coef", key="kl_ctrl_coef", format="%.1e")

    def _set_actor_ppo_micro_batch_size_per_gpu(self):
        st.number_input(
            "Micro Batch Size Per GPU for Actor",
            key="actor_ppo_micro_batch_size_per_gpu",
            min_value=1,
        )

    def _set_ref_log_prob_micro_batch_size_per_gpu(self):
        st.number_input(
            "Micro Batch Size Per GPU for Ref",
            key="ref_log_prob_micro_batch_size_per_gpu",
            min_value=1,
        )

    def _set_actor_ulysses_sequence_parallel_size(self):
        st.number_input(
            "Ulysses Sequence Parallel Size",
            key="actor_ulysses_sequence_parallel_size",
            min_value=1,
            max_value=8,
        )

    def _set_actor_lr(self):
        st.number_input(
            "Learning Rate for Actor",
            key="actor_lr",
            min_value=1e-7,
            max_value=1e-3,
            format="%.1e",
        )

    def _set_actor_warmup_style(self):
        st.selectbox(
            "LR Warmup Style for Actor",
            ["constant", "cosine"],
            key="actor_warmup_style",
        )

    def _set_actor_lr_warmup_steps_ratio(self):
        st.number_input(
            "LR Warmup Steps Ratio for Actor",
            key="actor_lr_warmup_steps_ratio",
            min_value=0.0,
            max_value=1.0,
        )

    def _set_actor_grad_clip(self):
        st.number_input("Grad Clip", key="actor_grad_clip", min_value=0.0, max_value=1.0)

    def _set_actor_clip_ratio(self):
        st.number_input("Clip Ratio", key="actor_clip_ratio", min_value=0.0, max_value=1.0)

    def _set_actor_entropy_coeff(self):
        st.number_input(
            "Entropy Coeff",
            key="actor_entropy_coeff",
            min_value=0.0,
            max_value=1.0,
            format="%.1e",
        )

    def _set_actor_use_kl_loss(self):
        st.session_state["actor_use_kl_loss"] = (
            st.session_state["algorithm_type"] == "ppo"
            and st.session_state["adv_estimator"] == "grpo"
        )  # TODO: check it

    def _set_actor_kl_loss_coef(self):
        st.number_input(
            "KL Loss Coef",
            key="actor_kl_loss_coef",
            min_value=0.0,
            max_value=1.0,
            format="%.1e",
            help="Used in advantage calcuation for GRPO",
        )

    def _set_actor_kl_loss_type(self):
        st.selectbox(
            "KL Loss Type",
            ["kl", "abs", "mse", "low_var_kl"],
            key="actor_kl_loss_type",
            help="Used in advantage calcuation for GRPO",
        )

    def _set_actor_tau(self):
        st.number_input(
            "Tau for OPMD",
            key="actor_tau",
            min_value=0.0,
            max_value=1.0,
            format="%.1e",
        )

    def _set_actor_opmd_baseline(self):
        st.selectbox(
            "OPMD Baseline",
            ["mean", "logavgexp"],
            key="actor_opmd_baseline",
        )

    def _set_actor_use_uid(self):
        st.checkbox("Use UID for OPMD", key="actor_use_uid")

    def _set_actor_checkpoint(self):
        st.multiselect(
            "Checkpoint",
            ["model", "hf_model", "optimizer", "extra"],
            key="actor_checkpoint",
        )

    def _set_critic_ppo_micro_batch_size_per_gpu(self):
        st.number_input(
            "Micro Batch Size Per GPU for Critic",
            key="critic_ppo_micro_batch_size_per_gpu",
            min_value=1,
        )

    def _set_critic_ulysses_sequence_parallel_size(self):
        st.number_input(
            "Ulysses Sequence Parallel Size",
            key="critic_ulysses_sequence_parallel_size",
            min_value=1,
            max_value=8,
        )

    def _set_critic_lr(self):
        st.number_input(
            "Learning Rate for Critic",
            key="critic_lr",
            min_value=1e-7,
            max_value=1e-3,
            format="%.1e",
        )

    def _set_critic_warmup_style(self):
        st.selectbox(
            "LR Warmup Style for Critic",
            ["constant", "cosine"],
            key="critic_warmup_style",
        )

    def _set_critic_lr_warmup_steps_ratio(self):
        st.number_input(
            "LR Warmup Steps Ratio for Critic",
            key="critic_lr_warmup_steps_ratio",
            min_value=0.0,
            max_value=1.0,
        )

    def _set_critic_grad_clip(self):
        st.number_input(
            "Grad Clip for Critic",
            key="critic_grad_clip",
            min_value=0.0,
            max_value=1.0,
        )

    def _set_critic_cliprange_value(self):
        st.number_input(
            "Cliprange Value",
            key="critic_cliprange_value",
            min_value=0.0,
            max_value=1.0,
        )

    def _set_training_mode(self):
        st.selectbox("Training Mode", ["PPO", "GRPO", "DPO", "OPMD"], key="training_mode")

        if st.session_state["training_mode"] == "PPO":
            st.session_state["algorithm_type"] = AlgorithmType.PPO.value
            st.session_state["adv_estimator"] = "gae"
        elif st.session_state["training_mode"] == "GRPO":
            st.session_state["algorithm_type"] = AlgorithmType.PPO.value
            st.session_state["adv_estimator"] = "grpo"
        elif st.session_state["training_mode"] == "DPO":
            st.session_state["algorithm_type"] = AlgorithmType.DPO.value
            st.session_state["adv_estimator"] = "grpo"
        elif st.session_state["training_mode"] == "OPMD":
            st.session_state["algorithm_type"] = AlgorithmType.OPMD.value
            st.session_state["adv_estimator"] = "grpo"

    def _set_configs_with_st_columns(
        self, config_names: List[str], columns_config: List[int] = None
    ):
        if columns_config is None:
            columns_config = len(config_names)
        columns = st.columns(columns_config)
        for col, config_name in zip(columns, config_names):
            with col:
                getattr(self, f"_set_{config_name}")()

    def beginner_mode(self):
        st.header("Essential Configs")
        self._set_configs_with_st_columns(["project", "name"], columns_config=[1, 3])

        self._set_model_path()

        self._set_checkpoint_path()

        self._set_dataset_path()

        self._set_configs_with_st_columns(["training_mode", "sft_warmup_iteration"])
        if st.session_state["sft_warmup_iteration"] > 0:
            self._set_sft_warmup_dataset_path()

        st.header("Important Configs")
        self._set_configs_with_st_columns(
            ["node_num", "gpu_per_node", "engine_num", "tensor_parallel_size"]
        )
        self._check_engine_num_and_tp_size()

        self._set_configs_with_st_columns(
            ["total_epoch", "task_num_per_batch", "max_prompt_tokens", "max_response_tokens"]
        )
        self._check_task_num_per_batch()

        self._set_dataset_args()

        if st.session_state["sft_warmup_iteration"] > 0:
            self._set_sft_warmup_dataset_args()

        self._set_configs_with_st_columns(["default_workflow_type", "default_reward_fn_type"])

        self._set_configs_with_st_columns(["sync_iteration_interval", "eval_interval", "save_freq"])

        self._set_actor_use_kl_loss()
        if st.session_state["actor_use_kl_loss"]:
            self._set_configs_with_st_columns(["actor_kl_loss_coef", "actor_kl_loss_type"])

        self._set_configs_with_st_columns(
            [
                "actor_ppo_micro_batch_size_per_gpu",
                "actor_lr",
                "ref_log_prob_micro_batch_size_per_gpu",
            ]
        )

        use_critic = (
            st.session_state["adv_estimator"] == "gae"
        )  # TODO: check it and may apply to expert mode
        if use_critic:
            self._set_configs_with_st_columns(["critic_ppo_micro_batch_size_per_gpu", "critic_lr"])

    def _expert_model_part(self):
        self._set_configs_with_st_columns(["project", "name"], columns_config=[1, 3])

        self._set_model_path()
        self._set_critic_model_path()

        self._set_checkpoint_path()

        self._set_configs_with_st_columns(
            ["node_num", "gpu_per_node", "max_prompt_tokens", "max_response_tokens"]
        )

    def _expert_buffer_part(self):
        self._set_configs_with_st_columns(["total_epoch", "task_num_per_batch"])
        self._check_task_num_per_batch()

        self._set_dataset_path()

        self._set_dataset_args()

        self._set_configs_with_st_columns(
            ["default_workflow_type", "default_reward_fn_type", "storage_type"]
        )

        self.buffer_advanced_tab = st.expander("Advanced Config")
        with self.buffer_advanced_tab:
            self._set_db_url()

            self._set_configs_with_st_columns(["max_retry_times", "max_retry_interval"])

            self._set_sft_warmup_dataset_path()
            self._set_sft_warmup_dataset_args()

    def _expert_connector_part(self):
        self._set_configs_with_st_columns(
            ["engine_type", "engine_num", "tensor_parallel_size", "repeat_times"]
        )
        self._check_engine_num_and_tp_size()

        self._set_configs_with_st_columns(["sync_method", "sync_iteration_interval"])

        with st.expander("Advanced Config"):
            self._set_configs_with_st_columns(
                ["runner_num", "max_pending_requests", "max_waiting_steps", "dtype"]
            )

            self._set_configs_with_st_columns(
                ["backend", "temperature", "top_p", "top_k", "seed", "logprobs"]
            )

            self._set_configs_with_st_columns(["enable_prefix_caching", "enforce_eager"])

    def _expert_trainer_part(self):
        self._set_configs_with_st_columns(
            ["trainer_type", "algorithm_type", "sft_warmup_iteration", "eval_interval"]
        )
        self._check_sft_warmup_dataset_path()

        if st.session_state["trainer_type"] == "verl":
            self._set_trainer_config_path()

            rl_training_tab, rl_algorithm_tab, actor_ref_tab, critic_tab = st.tabs(
                [
                    "RL Training Config",
                    "RL Algorithm Config",
                    "Actor and Ref Config",
                    "Critic Config",
                ]
            )
            with rl_training_tab:
                st.subheader("RL Training Config")
                self._set_training_args()

                self._set_configs_with_st_columns(["save_freq", "training_strategy", "resume_mode"])

                if st.session_state["training_strategy"] == "fsdp":
                    self._set_configs_with_st_columns(["param_offload", "optimizer_offload"])
                self._set_resume_from_path()

                with st.expander("Advanced Config"):
                    self._set_configs_with_st_columns(["critic_warmup", "total_training_steps"])

                    self._set_default_hdfs_dir()

                    self._set_configs_with_st_columns(
                        ["remove_previous_ckpt_in_save", "del_local_ckpt_after_load"]
                    )

                    self._set_configs_with_st_columns(
                        ["max_actor_ckpt_to_keep", "max_critic_ckpt_to_keep"]
                    )

            with rl_algorithm_tab:
                st.subheader("RL Algorithm Config")
                self._set_configs_with_st_columns(["gamma", "lam", "adv_estimator"])

                self._set_configs_with_st_columns(["kl_penalty", "kl_ctrl_type", "kl_ctrl_coef"])

            with actor_ref_tab:
                st.subheader("Actor Model Config")
                self._set_configs_with_st_columns(
                    [
                        "actor_ppo_micro_batch_size_per_gpu",
                        "ref_log_prob_micro_batch_size_per_gpu",
                        "actor_ulysses_sequence_parallel_size",
                    ]
                )

                self._set_configs_with_st_columns(
                    ["actor_lr", "actor_warmup_style", "actor_lr_warmup_steps_ratio"]
                )

                self._set_configs_with_st_columns(
                    ["actor_grad_clip", "actor_clip_ratio", "actor_entropy_coeff"]
                )

                self._set_actor_use_kl_loss()
                if st.session_state["actor_use_kl_loss"]:
                    self._set_configs_with_st_columns(["actor_kl_loss_coef", "actor_kl_loss_type"])

                if st.session_state["algorithm_type"] == "opmd":
                    self._set_configs_with_st_columns(
                        ["actor_tau", "actor_opmd_baseline", "actor_use_uid"]
                    )

                self._set_actor_checkpoint()

            with critic_tab:
                st.subheader("Critic Model Config")
                self._set_configs_with_st_columns(
                    ["critic_ppo_micro_batch_size_per_gpu", "critic_ulysses_sequence_parallel_size"]
                )

                self._set_configs_with_st_columns(
                    ["critic_lr", "critic_warmup_style", "critic_lr_warmup_steps_ratio"]
                )

                self._set_configs_with_st_columns(["critic_grad_clip", "critic_cliprange_value"])

    def expert_mode(self):
        model_tab, buffer_tab, connector_tab, trainer_tab = st.tabs(
            ["Model", "Data", "Explorer and Synchronizer", "Trainer"]
        )
        with model_tab:
            self._expert_model_part()

        with buffer_tab:
            self._expert_buffer_part()

        with connector_tab:
            self._expert_connector_part()

        with trainer_tab:
            self._expert_trainer_part()

    def generate_config(self):
        trainer_nnodes = (
            st.session_state["node_num"]
            - st.session_state["engine_num"]
            * st.session_state["tensor_parallel_size"]
            // st.session_state["gpu_per_node"]
        )
        if st.session_state["node_num"] == 1:
            trainer_n_gpus_per_node = (
                st.session_state["gpu_per_node"]
                - st.session_state["engine_num"] * st.session_state["tensor_parallel_size"]
            )
        else:
            trainer_n_gpus_per_node = st.session_state["gpu_per_node"]

        if st.session_state["trainer_type"] == "verl":
            balance_batch = "balance_batch" in st.session_state["training_args"]
            enable_gradient_checkpointing = (
                "gradient_checkpointing" in st.session_state["training_args"]
            )
            use_remove_padding = "remove_padding" in st.session_state["training_args"]
            use_dynamic_bsz = "dynamic_bsz" in st.session_state["training_args"]

            if st.session_state["training_strategy"] == "fsdp":
                fsdp_config = {
                    "wrap_policy": {"min_num_params": 0},
                    "param_offload": st.session_state["param_offload"],
                    "optimizer_offload": st.session_state["optimizer_offload"],
                    "fsdp_size": -1,
                }
            else:
                fsdp_config = {}
            trainer_config = {
                "data": {
                    "tokenizer": None,
                    "train_files": "placeholder",
                    "val_files": "placeholder",
                    "prompt_key": "placeholder",
                    "max_prompt_length": st.session_state["max_prompt_tokens"],
                    "max_response_length": st.session_state["max_response_tokens"],
                    "train_batch_size": st.session_state["task_num_per_batch"]
                    * st.session_state["repeat_times"],
                    "val_batch_size": None,
                    "return_raw_input_ids": False,
                    "return_raw_chat": False,
                    "shuffle": True,
                    "filter_overlong_prompts": False,
                    "truncation": "error",
                    "image_key": "images",
                },
                "actor_rollout_ref": {
                    "hybrid_engine": True,
                    "model": {
                        "path": st.session_state["model_path"],
                        "external_lib": None,
                        "override_config": {},
                        "enable_gradient_checkpointing": enable_gradient_checkpointing,
                        "use_remove_padding": use_remove_padding,
                    },
                    "actor": {
                        "strategy": st.session_state["training_strategy"],
                        "ppo_mini_batch_size": st.session_state["task_num_per_batch"],
                        "ppo_micro_batch_size_per_gpu": st.session_state[
                            "actor_ppo_micro_batch_size_per_gpu"
                        ],
                        "use_dynamic_bsz": use_dynamic_bsz,
                        "ppo_max_token_len_per_gpu": st.session_state["repeat_times"]
                        * (
                            st.session_state["max_prompt_tokens"]
                            + st.session_state["max_response_tokens"]
                        ),
                        "grad_clip": st.session_state["actor_grad_clip"],
                        "clip_ratio": st.session_state["actor_clip_ratio"],
                        "entropy_coeff": st.session_state["actor_entropy_coeff"],
                        "use_kl_loss": st.session_state["actor_use_kl_loss"],
                        "kl_loss_coef": st.session_state["actor_kl_loss_coef"],
                        "kl_loss_type": st.session_state["actor_kl_loss_type"],
                        "ppo_epochs": 1,  # TODO
                        "shuffle": False,
                        "ulysses_sequence_parallel_size": st.session_state[
                            "actor_ulysses_sequence_parallel_size"
                        ],
                        "checkpoint": {"contents": st.session_state["actor_checkpoint"]},
                        "optim": {
                            "lr": st.session_state["actor_lr"],
                            "lr_warmup_steps_ratio": st.session_state[
                                "actor_lr_warmup_steps_ratio"
                            ],
                            "warmup_style": st.session_state["actor_warmup_style"],
                            "total_training_steps": -1
                            if st.session_state["total_training_steps"] is None
                            else st.session_state["total_training_steps"],
                        },
                        "fsdp_config": fsdp_config,
                        "alg_type": st.session_state["algorithm_type"],
                        "tau": st.session_state["actor_tau"],
                        "opmd_baseline": st.session_state["actor_opmd_baseline"],
                        "use_uid": st.session_state["actor_use_uid"],
                    },
                    "ref": {
                        "fsdp_config": fsdp_config,
                        "log_prob_micro_batch_size_per_gpu": st.session_state[
                            "ref_log_prob_micro_batch_size_per_gpu"
                        ],
                        "log_prob_use_dynamic_bsz": "${actor_rollout_ref.actor.use_dynamic_bsz}",
                        "log_prob_max_token_len_per_gpu": "${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}",
                        "ulysses_sequence_parallel_size": "${actor_rollout_ref.actor.ulysses_sequence_parallel_size}",
                    },
                    "rollout": {
                        "name": "vllm",
                        "temperature": st.session_state["temperature"],
                        "top_k": -1,
                        "top_p": 1,
                        "use_fire_sampling": False,
                        "prompt_length": "${data.max_prompt_length}",
                        "response_length": "${data.max_response_length}",
                        "dtype": "bfloat16",
                        "gpu_memory_utilization": 0.4,
                        "ignore_eos": False,
                        "enforce_eager": True,
                        "free_cache_engine": True,
                        "load_format": "dummy_dtensor",
                        "tensor_model_parallel_size": 2,
                        "max_num_batched_tokens": 8192,
                        "max_model_len": None,
                        "max_num_seqs": 1024,
                        "log_prob_micro_batch_size_per_gpu": 4,
                        "log_prob_use_dynamic_bsz": "${actor_rollout_ref.actor.use_dynamic_bsz}",
                        "log_prob_max_token_len_per_gpu": "${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}",
                        "disable_log_stats": True,
                        "enable_chunked_prefill": True,
                        "do_sample": True,
                        "n": st.session_state["repeat_times"],
                    },
                },
                "critic": {
                    "strategy": st.session_state["training_strategy"],
                    "optim": {
                        "lr": st.session_state["critic_lr"],
                        "lr_warmup_steps_ratio": st.session_state["critic_warmup_style"],
                        "warmup_style": st.session_state["critic_lr_warmup_steps_ratio"],
                        "total_training_steps": -1
                        if st.session_state["total_training_steps"] is None
                        else st.session_state["total_training_steps"],
                    },
                    "model": {
                        "path": st.session_state["critic_model_path"]
                        if st.session_state["critic_model_path"].strip()
                        else st.session_state["model_path"],
                        "tokenizer_path": "${actor_rollout_ref.model.path}",
                        "override_config": {},
                        "external_lib": "${actor_rollout_ref.model.external_lib}",
                        "enable_gradient_checkpointing": enable_gradient_checkpointing,
                        "use_remove_padding": use_remove_padding,
                        "fsdp_config": fsdp_config,
                    },
                    "ppo_mini_batch_size": "${actor_rollout_ref.actor.ppo_mini_batch_size}",
                    "ppo_micro_batch_size_per_gpu": st.session_state[
                        "critic_ppo_micro_batch_size_per_gpu"
                    ],
                    "forward_micro_batch_size_per_gpu": "${critic.ppo_micro_batch_size_per_gpu}",
                    "use_dynamic_bsz": use_dynamic_bsz,
                    "ppo_max_token_len_per_gpu": st.session_state["repeat_times"]
                    * (
                        st.session_state["max_prompt_tokens"]
                        + st.session_state["max_response_tokens"]
                    )
                    * 2,
                    "forward_max_token_len_per_gpu": "${critic.ppo_max_token_len_per_gpu}",
                    "ulysses_sequence_parallel_size": st.session_state[
                        "critic_ulysses_sequence_parallel_size"
                    ],
                    "ppo_epochs": "${actor_rollout_ref.actor.ppo_epochs}",
                    "shuffle": "${actor_rollout_ref.actor.shuffle}",
                    "grad_clip": st.session_state["critic_grad_clip"],
                    "cliprange_value": st.session_state["critic_cliprange_value"],
                },
                "reward_model": {
                    "enable": False,
                    "strategy": "fsdp",
                    "model": {
                        "input_tokenizer": "${actor_rollout_ref.model.path}",
                        "path": "~/models/FsfairX-LLaMA3-RM-v0.1",
                        "external_lib": "${actor_rollout_ref.model.external_lib}",
                        "use_remove_padding": False,
                        "fsdp_config": {
                            "min_num_params": 0,
                            "param_offload": False,
                            "fsdp_size": -1,
                        },
                    },
                    "ulysses_sequence_parallel_size": 1,
                    "use_dynamic_bsz": "${critic.use_dynamic_bsz}",
                    "forward_max_token_len_per_gpu": "${critic.forward_max_token_len_per_gpu}",
                    "reward_manager": "naive",
                },
                "custom_reward_function": {"path": None, "name": "compute_score"},
                "algorithm": {
                    "gamma": st.session_state["gamma"],
                    "lam": st.session_state["lam"],
                    "adv_estimator": st.session_state["adv_estimator"],
                    "kl_penalty": st.session_state["kl_penalty"],
                    "kl_ctrl": {
                        "type": st.session_state["kl_ctrl_type"],
                        "kl_coef": st.session_state["kl_ctrl_coef"],
                    },
                },
                "trainer": {
                    "balance_batch": balance_batch,
                    "total_epochs": st.session_state["total_epoch"],
                    "project_name": st.session_state["project"],
                    "experiment_name": st.session_state["exp_name"],
                    "logger": ["wandb"],
                    "val_generations_to_log_to_wandb": 0,
                    "nnodes": trainer_nnodes,
                    "n_gpus_per_node": trainer_n_gpus_per_node,
                    "save_freq": st.session_state["save_freq"],
                    "resume_mode": st.session_state["resume_mode"],
                    "resume_from_path": st.session_state["resume_from_path"],
                    "test_freq": 100,
                    "critic_warmup": st.session_state["critic_warmup"],
                    "default_hdfs_dir": st.session_state["default_hdfs_dir"],
                    "remove_previous_ckpt_in_save": st.session_state[
                        "remove_previous_ckpt_in_save"
                    ],
                    "del_local_ckpt_after_load": st.session_state["del_local_ckpt_after_load"],
                    "default_local_dir": st.session_state["checkpoint_path"],
                    "val_before_train": False,
                    "sync_freq": st.session_state["sync_iteration_interval"],
                    "max_actor_ckpt_to_keep": st.session_state["max_actor_ckpt_to_keep"],
                    "max_critic_ckpt_to_keep": st.session_state["max_critic_ckpt_to_keep"],
                },
            }
        else:
            raise ValueError(f"Invalid trainer type: {st.session_state['trainer_type']}")

        if len(self.unfinished_fields) > 0:
            disable_generate = True
            help_messages = (
                f"Please check following fields: `{'`, `'.join(self.unfinished_fields)}`"
            )
        else:
            disable_generate = False
            help_messages = None
        if st.button(
            "Generate Config",
            disabled=disable_generate,
            help=help_messages,
        ):
            config = {
                "data": {
                    "total_epochs": st.session_state["total_epoch"],
                    "batch_size": st.session_state["task_num_per_batch"],
                    "dataset_path": st.session_state["dataset_path"],
                    "default_workflow_type": st.session_state["default_workflow_type"],
                    "default_reward_fn_type": st.session_state["default_reward_fn_type"],
                    "train_split": st.session_state["train_split"],
                    "eval_split": st.session_state["eval_split"],
                    "format_config": {
                        "prompt_key": st.session_state["prompt_key"],
                        "response_key": st.session_state["response_key"],
                    },
                },
                "model": {
                    "model_path": st.session_state["model_path"],
                    "max_prompt_tokens": st.session_state["max_prompt_tokens"],
                    "max_response_tokens": st.session_state["max_response_tokens"],
                    "checkpoint_path": st.session_state["checkpoint_path"],
                },
                "cluster": {
                    "node_num": st.session_state["node_num"],
                    "gpu_per_node": st.session_state["gpu_per_node"],
                },
                "buffer": {
                    "storage_type": st.session_state["storage_type"],
                    "db_url": st.session_state["db_url"]
                    if st.session_state["db_url"].strip()
                    else f"sqlite:///{os.path.join(st.session_state['checkpoint_path'], '.cache', st.session_state['project'], st.session_state['exp_name'])}/data.db",
                    "read_batch_size": st.session_state["task_num_per_batch"]
                    * st.session_state["repeat_times"],
                    "max_retry_times": st.session_state["max_retry_times"],
                    "max_retry_interval": st.session_state["max_retry_interval"],
                },
                "explorer": {
                    "engine_type": st.session_state["engine_type"],
                    "engine_num": st.session_state["engine_num"],
                    "runner_num": st.session_state["runner_num"],
                    "tensor_parallel_size": st.session_state["tensor_parallel_size"],
                    "enable_prefix_caching": st.session_state["enable_prefix_caching"],
                    "enforce_eager": st.session_state["enforce_eager"],
                    "dtype": st.session_state["dtype"],
                    "temperature": st.session_state["temperature"],
                    "top_p": st.session_state["top_p"],
                    "top_k": st.session_state["top_k"],
                    "seed": st.session_state["seed"],
                    "logprobs": st.session_state["logprobs"],
                    "repeat_times": st.session_state["repeat_times"],
                    "backend": st.session_state["backend"],
                    "max_pending_requests": st.session_state["max_pending_requests"],
                    "max_waiting_steps": st.session_state["max_waiting_steps"],
                },
                "synchronizer": {
                    "sync_method": st.session_state["sync_method"],
                    "sync_iteration_interval": st.session_state["sync_iteration_interval"],
                },
                "trainer": {
                    "trainer_type": st.session_state["trainer_type"],
                    "algorithm_type": st.session_state["algorithm_type"],
                    "trainer_config_path": st.session_state["trainer_config_path"],
                    "sft_warmup_iteration": st.session_state["sft_warmup_iteration"],
                    "eval_interval": st.session_state["eval_interval"],
                },
                "monitor": {
                    "project": st.session_state["project"],
                    "name": st.session_state["exp_name"],
                },
            }
            st.header("Generated Config File")
            st.subheader("Overall Config File")
            yaml_config = yaml.dump(config, allow_unicode=True, sort_keys=False)
            st.code(yaml_config, language="yaml")
            st.subheader("Trainer Config File")
            trainer_config = yaml.dump(trainer_config, allow_unicode=True, sort_keys=False)
            st.code(trainer_config, language="yaml")


if __name__ == "__main__":
    config_manager = ConfigManager()
