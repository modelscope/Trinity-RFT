import copy
import os
import subprocess
import tempfile
from typing import List

import streamlit as st
import yaml

from trinity.common.constants import (
    AlgorithmType,
    MonitorType,
    PromptType,
    StorageType,
    SyncMethod,
)
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows.workflow import WORKFLOWS
from trinity.trainer.verl.ray_trainer import AdvantageEstimator


class ConfigManager:
    def __init__(self):
        self._init_default_config()
        self.unfinished_fields = set()
        st.set_page_config(page_title="Trinity-RFT Config Generator", page_icon=":robot:")
        st.title("Trinity-RFT Config Generator")
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
        if "config_generated" not in st.session_state:
            st.session_state.config_generated = False
        if "is_running" not in st.session_state:
            st.session_state.is_running = False
        self.generate_config()

    def _init_default_config(self):
        self.default_config = {
            "_init_config_manager": True,
            "mode": "both",
            "project": "Trinity-RFT",
            "exp_name": "qwen2.5-1.5B",
            "checkpoint_root_dir": "",
            "monitor_type": MonitorType.TENSORBOARD.value,
            # Algorithm Configs
            "algorithm_type": AlgorithmType.PPO.value,
            "_grouped_adv_repeat_times": 2,
            "_not_grouped_adv_repeat_times": 1,
            "repeat_times": 1,
            "gamma": 1.0,
            "lam": 1.0,
            # Model Configs
            "model_path": "",
            "critic_model_path": "",
            "max_prompt_tokens": 1024,
            "max_response_tokens": 1024,
            # Cluster Config
            "node_num": 1,
            "gpu_per_node": 8,
            "total_gpu_num": 8,
            "trainer_gpu_num": 6,
            # Buffer Configs
            "total_epochs": 20,
            "_train_batch_size_per_gpu": 16,
            "train_batch_size": 96,
            "buffer_max_retry_times": 3,
            "max_retry_interval": 1,
            # Taskset Configs
            "taskset_path": "",
            "taskset_subset_name": None,
            "taskset_split": "train",
            "taskset_prompt_key": "question",
            "taskset_response_key": "answer",
            "temperature": 1.0,
            "top_p": 1.0,  # TODO: to be used
            "top_k": -1,  # TODO: to be used
            "logprobs": 0,
            # Eval Taskset Configs
            "_eval_tasksets_num": 0,
            # Explorer Input Configs
            "default_workflow_type": "math_workflow",
            "default_reward_fn_type": "math_reward",
            "system_prompt": None,
            "reply_prefix": None,
            # Experience Buffer / DPO Dataset Configs
            "_dpo_storage_type": StorageType.FILE.value,
            "_not_dpo_storage_type": StorageType.QUEUE.value,
            "storage_type": StorageType.QUEUE.value,
            "_dpo_experience_buffer_path": "",
            "_not_dpo_experience_buffer_path": "",
            "experience_buffer_path": "",
            "dpo_dataset_train_split": "train",
            "dpo_dataset_prompt_type": PromptType.MESSAGES.value,
            "dpo_dataset_prompt_key": "prompt",
            "dpo_dataset_chosen_key": "chosen",
            "dpo_dataset_rejected_key": "rejected",
            # SFT Warmup Dataset Configs
            "sft_warmup_dataset_path": "",
            "sft_warmup_train_split": "train",
            "sft_warmup_prompt_type": PromptType.MESSAGES.value,
            "sft_warmup_messages_key": "messages",
            "sft_warmup_prompt_key": "prompt",
            "sft_warmup_response_key": "response",
            # TrainerInput Configs
            # TODO: read_experience_strategy
            "sft_warmup_steps": 0,
            # Explorer and Sync Configs
            "runner_num": 32,
            "max_timeout": 900,
            "explorer_max_retry_times": 2,
            "eval_interval": 1000,
            "eval_on_latest_checkpoint": True,
            # Rollout Model Configs
            "engine_type": "vllm_async",
            "engine_num": 2,
            "tensor_parallel_size": 1,
            "use_v1": True,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "gpu_memory_utilization": 0.9,
            "dtype": "bfloat16",
            "seed": 42,
            # TODO: max_prompt_tokens
            # TODO: max_response_tokens
            # TODO: chat_template
            "enable_thinking": False,
            "enable_openai_api": False,
            # TODO: Auxiliary Models Configs
            # Synchronizer Configs
            "_not_dpo_sync_method": SyncMethod.NCCL.value,
            "sync_method": SyncMethod.NCCL.value,
            "sync_interval": 10,
            "sync_timeout": 1200,
            # Trainer Configs
            "trainer_type": "verl",
            "_nccl_save_interval": 100,
            "save_interval": 100,
            # TODO: enable_preview
            "_not_dpo_actor_use_kl_loss": True,
            "actor_use_kl_loss": True,
            "actor_kl_loss_coef": 0.001,
            "actor_entropy_coef": 0.001,
            "actor_grad_clip": 1.0,
            "actor_clip_ratio": 0.2,
            # veRL Trainer Configs
            "training_args": [
                "balance_batch",
                "gradient_checkpointing",
                "remove_padding",
                "dynamic_bsz",
            ],
            "ppo_epochs": 1,
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
            "adv_estimator": "gae",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "kl_penalty": "low_var_kl",
            "kl_ctrl_type": "fixed",
            "kl_ctrl_coef": 0.001,
            "horizon": 10000,
            "target_kl": 0.1,
            "actor_ppo_micro_batch_size_per_gpu": 4,
            "ref_log_prob_micro_batch_size_per_gpu": 8,
            "actor_ulysses_sequence_parallel_size": 1,
            "actor_lr": 1e-6,
            "actor_warmup_style": "constant",
            "actor_lr_warmup_steps_ratio": 0.0,
            "actor_tau": 0.0,
            "actor_opmd_baseline": "mean",
            "actor_use_uid": False,
            "actor_kl_loss_type": "low_var_kl",
            "actor_checkpoint": ["model", "hf_model", "optimizer", "extra"],
            "critic_lr": 1e-6,
            "critic_warmup_style": "constant",
            "critic_lr_warmup_steps_ratio": 0.0,
            "critic_grad_clip": 1.0,
            "critic_cliprange_value": 0.5,
            "critic_ppo_micro_batch_size_per_gpu": 8,
            "critic_ulysses_sequence_parallel_size": 1,
            "critic_checkpoint": ["model", "optimizer", "extra"],
        }

    def reset_session_state(self):
        for key, value in self.default_config.items():
            st.session_state[key] = value

    def maintain_session_state(self):
        for key in self.default_config:
            st.session_state[key] = st.session_state[key]
        eavl_dataset_keys = ["name", "path", "subset_name", "split", "prompt_key", "response_key"]
        for idx in range(st.session_state["_eval_tasksets_num"]):
            for key in eavl_dataset_keys:
                full_key = f"eval_taskset_{idx}_{key}"
                st.session_state[full_key] = st.session_state[full_key]

    def _set_project(self):
        st.text_input("Project", key="project")

    def _set_exp_name(self):
        st.text_input("Experiment Name", key="exp_name")

    def _set_monitor_type(self):
        st.selectbox(
            "Monitor Type",
            options=[monitor_type.value for monitor_type in MonitorType],
            key="monitor_type",
        )

    def _set_model_path(self):
        st.text_input("Model Path", key="model_path")
        if not st.session_state["model_path"].strip():
            self.unfinished_fields.add("model_path")
            st.warning("Please input model path.")

    def _set_critic_model_path(self):
        if st.session_state["adv_estimator"] == AdvantageEstimator.GAE.value:
            st.text_input(
                "Critic Model Path (defaults to `model_path`)",
                key="critic_model_path",
            )

    def _set_checkpoint_root_dir(self):
        st.text_input("Checkpoint Root Dir", key="checkpoint_root_dir")
        if not st.session_state["checkpoint_root_dir"].strip():  # TODO: may auto generate
            self.unfinished_fields.add("checkpoint_root_dir")
            st.warning("Please input checkpoint root dir.")
        elif not os.path.isabs(st.session_state["checkpoint_root_dir"].strip()):
            self.unfinished_fields.add("checkpoint_root_dir")
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
        if st.session_state["mode"] == "both":
            st.session_state["trainer_gpu_num"] = (
                st.session_state["total_gpu_num"]
                - st.session_state["engine_num"] * st.session_state["tensor_parallel_size"]
            )
        else:  # model == train
            st.session_state["trainer_gpu_num"] = st.session_state["total_gpu_num"]

    def _set_max_prompt_tokens(self):
        st.number_input("Max Prompt Tokens", key="max_prompt_tokens", min_value=1)

    def _set_max_response_tokens(self):
        st.number_input("Max Response Tokens", key="max_response_tokens", min_value=1)

    def _set_total_epochs(self):
        st.number_input("Total Epochs", key="total_epochs", min_value=1)

    @property
    def _str_for_train_batch_size(self):
        trainer_gpu_num_str = (
            "`gpu_per_node * node_num - engine_num * tensor_parallel_size`"
            if st.session_state["mode"] == "both"
            else "`gpu_per_node * node_num`"
        )
        return (
            f"Please ensure that `train_batch_size` can be divided by "
            f"{trainer_gpu_num_str} = {st.session_state['trainer_gpu_num']}."
        )

    def _set_train_batch_size(self):
        trainer_gpu_num = st.session_state["trainer_gpu_num"]
        st.session_state["train_batch_size"] = (
            st.session_state["_train_batch_size_per_gpu"] * st.session_state["trainer_gpu_num"]
        )

        def on_change():
            st.session_state["_train_batch_size_per_gpu"] = max(
                st.session_state["train_batch_size"] // st.session_state["trainer_gpu_num"], 1
            )

        st.number_input(
            "Train Batch Size",
            key="train_batch_size",
            min_value=trainer_gpu_num,
            step=trainer_gpu_num,
            help=self._str_for_train_batch_size,
            on_change=on_change,
        )

    def _check_train_batch_size(self):
        if st.session_state["train_batch_size"] % st.session_state["trainer_gpu_num"] != 0:
            self.unfinished_fields.add("train_batch_size")
            st.warning(self._str_for_train_batch_size)

    def _set_taskset_path(self):
        st.text_input("Taskset Path", key="taskset_path")
        if not st.session_state["taskset_path"].strip():
            self.unfinished_fields.add("taskset_path")
            st.warning("Please input taskset path.")

    def _set_system_prompt(self):
        st.text_area(
            "System Prompt",
            key="system_prompt",
            placeholder="System prompt is used to guide the model behavior.",
        )

    def _set_reply_prefix(self):
        st.text_area(
            "Assistant Reply Prefix",
            key="reply_prefix",
            placeholder="""Assistant reply prefix is used to specify the initial content of model reply, """
            """and a common setting is: \nLet me solve this step by step. """,
        )

    def _set_taskset_args(self):
        if st.session_state["taskset_path"] and "://" not in st.session_state["taskset_path"]:
            subset_name_col, split_col = st.columns(2)
            subset_name_col.text_input(
                "Subset Name :orange-badge[(Needs review)]",
                key="taskset_subset_name",
                help="The subset name used for `datasets.load_datasets`, see "
                "[here](https://huggingface.co/docs/datasets/v3.5.0/en/package_reference/loading_methods#datasets.load_dataset.name) for details.",
            )
            split_col.text_input("Train Split :orange-badge[(Needs review)]", key="taskset_split")
            prompt_key_col, response_key_col = st.columns(2)
            prompt_key_col.text_input(
                "Prompt Key :orange-badge[(Needs review)]", key="taskset_prompt_key"
            )
            response_key_col.text_input(
                "Response Key :orange-badge[(Needs review)]", key="taskset_response_key"
            )
            self._set_configs_with_st_columns(["temperature", "logprobs"])

    def _set_eval_taskset_idx(self, idx):  # TODO: add delete
        st.text_input(
            "Taskset Name",
            key=f"eval_taskset_{idx}_name",
        )
        st.text_input(
            "Eval Taskset Path",
            key=f"eval_taskset_{idx}_path",
        )
        if not st.session_state[f"eval_taskset_{idx}_path"].strip():
            st.warning("Please input the taskset path, or it will be ignored.")
        subset_name_col, split_col = st.columns(2)
        subset_name_col.text_input(
            "Subset Name :orange-badge[(Needs review)]",
            key=f"eval_taskset_{idx}_subset_name",
            help="The subset name used for `datasets.load_datasets`, see "
            "[here](https://huggingface.co/docs/datasets/v3.5.0/en/package_reference/loading_methods#datasets.load_dataset.name) for details.",
        )
        split_col.text_input(
            "Eval Split :orange-badge[(Needs review)]",
            key=f"eval_taskset_{idx}_split",
        )
        prompt_key_col, response_key_col = st.columns(2)
        prompt_key_col.text_input(
            "Prompt Key :orange-badge[(Needs review)]",
            key=f"eval_taskset_{idx}_prompt_key",
        )
        response_key_col.text_input(
            "Response Key :orange-badge[(Needs review)]",
            key=f"eval_taskset_{idx}_response_key",
        )

    def _set_eval_tasksets(self):
        if st.button("Add Eval Taskset"):
            st.session_state["_eval_tasksets_num"] += 1
        if st.session_state["_eval_tasksets_num"] > 0:
            tabs = st.tabs(
                [f"Eval Taskset {i + 1}" for i in range(st.session_state["_eval_tasksets_num"])]
            )
            for idx, tab in enumerate(tabs):
                with tab:
                    self._set_eval_taskset_idx(idx)

    def _set_default_workflow_type(self):
        st.selectbox(
            "Default Workflow Type :orange-badge[(Needs review)]",
            WORKFLOWS.modules.keys(),
            key="default_workflow_type",
            help=r"""`simple_workflow`: call 'model.chat()' to get responses.

`math_workflow`: call 'model.chat()' with a pre-defined system prompt to get responses.

Other workflows: conduct multi-turn task for the given dataset.
""",
        )

    def _set_default_reward_fn_type(self):
        st.selectbox(
            "Default Reward Fn Type :orange-badge[(Needs review)]",
            REWARD_FUNCTIONS.modules.keys(),
            key="default_reward_fn_type",
            help=r"""`accuracy_reward`: check the accuracy for math problems.

`format_reward`: check if the response matches the format (default: `<think>*</think>* <answer>*</answer>`).

`math_reward`: `accuracy_reward` (1 or 0) + `format_reward` (+0.1 or -0.1).
""",
        )

    def _set_storage_type(self):
        if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
            st.session_state["storage_type"] = st.session_state["_dpo_storage_type"]
            storage_candidates = [StorageType.FILE.value, StorageType.SQL.value]
        else:
            st.session_state["storage_type"] = st.session_state["_not_dpo_storage_type"]
            storage_candidates = [StorageType.QUEUE.value, StorageType.SQL.value]

        def on_change():
            if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
                st.session_state["_dpo_storage_type"] = st.session_state["storage_type"]
            else:
                st.session_state["_not_dpo_storage_type"] = st.session_state["storage_type"]

        st.selectbox(
            "Storage Type",
            storage_candidates,
            key="storage_type",
            on_change=on_change,
        )

    def _set_experience_buffer_path(self):  # TODO
        if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
            st.session_state["experience_buffer_path"] = st.session_state[
                "_dpo_experience_buffer_path"
            ]
            title = "DPO Dataset Path"
            help_msg = r"""This path to DPO dataset,

if `storage_type == StorageType.FILE`, this should be a path to a file,

if `storage_type == StorageType.SQL`, this should be a path to database."""
        else:
            st.session_state["experience_buffer_path"] = st.session_state[
                "_not_dpo_experience_buffer_path"
            ]
            title = "Experience Buffer Path"
            help_msg = r"""This path is used for `trainer`,

if `storage_type == StorageType.QUEUE`, default to `None`,

if `storage_type == StorageType.SQL`, default to `sqlite:///{os.path.join(checkpoint_root_dir, '.cache', project_name, experiment_name)}/data.db`."""

        def on_change():
            if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
                st.session_state["_dpo_experience_buffer_path"] = st.session_state[
                    "experience_buffer_path"
                ]
            else:
                st.session_state["_not_dpo_experience_buffer_path"] = st.session_state[
                    "experience_buffer_path"
                ]

        st.text_input(
            title,
            key="experience_buffer_path",
            help=help_msg,
            on_change=on_change,
        )
        if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
            if not st.session_state["experience_buffer_path"].strip():
                self.unfinished_fields.add("experience_buffer_path")
                st.warning("Please input DPO dataset path.")

    def _set_buffer_max_retry_times(self):
        st.number_input("Max Retry Times", key="buffer_max_retry_times", min_value=1)

    def _set_max_retry_interval(self):
        st.number_input("Max Retry Interval", key="max_retry_interval", min_value=1)

    def _set_dpo_dataset_kwargs(self):
        dpo_dataset_train_split_col, dpo_dataset_prompt_type_col = st.columns(2)
        dpo_dataset_train_split_col.text_input(
            "DPO Dataset Train Split :orange-badge[(Needs review)]", key="dpo_dataset_train_split"
        )
        dpo_dataset_prompt_type_col.selectbox(
            "DPO Dataset Prompt Type :orange-badge[(Needs review)]",
            [prompt_type.value for prompt_type in PromptType],
            key="dpo_dataset_prompt_type",
        )

        (
            dpo_dataset_prompt_key_col,
            dpo_dataset_chosen_key_col,
            dpo_dataset_rejected_key_col,
        ) = st.columns(3)
        dpo_dataset_prompt_key_col.text_input(
            "DPO Dataset Prompt Key :orange-badge[(Needs review)]", key="dpo_dataset_prompt_key"
        )
        dpo_dataset_chosen_key_col.text_input(
            "DPO Dataset Chosen Key :orange-badge[(Needs review)]", key="dpo_dataset_chosen_key"
        )
        dpo_dataset_rejected_key_col.text_input(
            "DPO Dataset Rejected Key :orange-badge[(Needs review)]",
            key="dpo_dataset_rejected_key",
        )

    def _check_sft_warmup_dataset_path(self):
        if st.session_state["sft_warmup_steps"]:
            if not st.session_state["sft_warmup_dataset_path"].strip():
                self.unfinished_fields.add("sft_warmup_dataset_path")
                st.warning("Please input SFT warmup dataset path when `sft_warmup_steps` is not 0")

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
                sft_warmup_prompt_type_col,
            ) = st.columns(2)
            sft_warmup_train_split_col.text_input(
                "SFT Dataset Train Split :orange-badge[(Needs review)]",
                key="sft_warmup_train_split",
            )
            sft_warmup_prompt_type_col.selectbox(
                "SFT Dataset Prompt Type :orange-badge[(Needs review)]",
                [prompt_type.value for prompt_type in PromptType],
                key="sft_warmup_prompt_type",
            )
            (
                sft_warmup_messages_key_col,
                sft_warmup_prompt_key_col,
                sft_warmup_response_key_col,
            ) = st.columns(
                3
            )  # TODO: select by prompt type
            sft_warmup_messages_key_col.text_input(
                "SFT Dataset Messages Key :orange-badge[(Needs review)]",
                key="sft_warmup_messages_key",
            )
            sft_warmup_prompt_key_col.text_input(
                "SFT Dataset Prompt Key :orange-badge[(Needs review)]", key="sft_warmup_prompt_key"
            )
            sft_warmup_response_key_col.text_input(
                "SFT Dataset Response Key :orange-badge[(Needs review)]",
                key="sft_warmup_response_key",
            )

    def _set_engine_type(self):
        st.selectbox("Explorer Engine Type", ["vllm_async", "vllm"], key="engine_type")

    @property
    def _str_for_engine_num_and_tp_size(self):
        return r"""and it must meet the following constraints:
```python
assert engine_num * tensor_parallel_size < gpu_per_node * node_num
if node_num > 1:
    assert gpu_per_node % tensor_parallel_size == 0
    assert engine_num * tensor_parallel_size % gpu_per_node == 0
```"""

    def _set_engine_num(self):
        total_gpu_num = st.session_state["total_gpu_num"]
        max_engine_num = (total_gpu_num - 1) // st.session_state["tensor_parallel_size"]
        if st.session_state["engine_num"] > max_engine_num:
            st.session_state["engine_num"] = max_engine_num
            self._set_trainer_gpu_num()
        st.number_input(
            "Engine Num",
            key="engine_num",
            min_value=1,
            max_value=max_engine_num,
            help=f"`engine_num` is used to set the quantity of inference engines, "
            f"{self._str_for_engine_num_and_tp_size}",
            on_change=self._set_trainer_gpu_num,
        )

    def _set_tensor_parallel_size(self):
        total_gpu_num = st.session_state["total_gpu_num"]
        max_tensor_parallel_size = (total_gpu_num - 1) // st.session_state["engine_num"]
        if st.session_state["tensor_parallel_size"] > max_tensor_parallel_size:
            st.session_state["tensor_parallel_size"] = max_tensor_parallel_size
            self._set_trainer_gpu_num()
        st.number_input(
            "Tensor Parallel Size",
            key="tensor_parallel_size",
            min_value=1,
            max_value=max_tensor_parallel_size,
            help=f"`tensor_parallel_size` is used to set the tensor parallel size of inference engines, "
            f"{self._str_for_engine_num_and_tp_size}",
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

    def _set_repeat_times(self):  # TODO
        grouped_adv_algorithms = [
            AlgorithmType.GRPO.value,
            AlgorithmType.OPMD.value,  # TODO: may add rloo
        ]
        if st.session_state["algorithm_type"] in grouped_adv_algorithms:
            min_repeat_times = 2
            st.session_state["repeat_times"] = st.session_state["_grouped_adv_repeat_times"]
        else:
            min_repeat_times = 1
            st.session_state["repeat_times"] = st.session_state["_not_grouped_adv_repeat_times"]

        def on_change():
            if st.session_state["algorithm_type"] in grouped_adv_algorithms:
                st.session_state["_grouped_adv_repeat_times"] = st.session_state["repeat_times"]
            else:
                st.session_state["_not_grouped_adv_repeat_times"] = st.session_state["repeat_times"]

        st.number_input(
            "Repeat Times",
            key="repeat_times",
            min_value=min_repeat_times,
            help="`repeat_times` is used to set how many experiences each task can generate, "
            "and it must be greater than `1` when `algorithm_type` is `opmd` or `grpo`.",
            on_change=on_change,
        )

    def _set_sync_method(self):
        if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
            st.session_state["sync_method"] = SyncMethod.CHECKPOINT.value
            disabled = True
        else:
            st.session_state["sync_method"] = st.session_state["_not_dpo_sync_method"]
            disabled = False

        def on_change():
            if st.session_state["algorithm_type"] != AlgorithmType.DPO.value:
                st.session_state["_not_dpo_sync_method"] = st.session_state["sync_method"]

        st.selectbox(
            "Sync Method",
            [sync_method.value for sync_method in SyncMethod],
            key="sync_method",
            help="""`nccl`: the explorer and trainer sync model weights once every `sync_interval` steps.

`checkpoint`: the trainer saves the model checkpoint, and the explorer loads it at `sync_interval`.""",
            disabled=disabled,
            on_change=on_change,
        )

    def _set_sync_interval(self):
        st.number_input(
            "Sync Interval",
            key="sync_interval",
            min_value=1,
            help="""The step interval at which the `explorer` and `trainer` synchronize model weight.""",
        )

    def _set_sync_timeout(self):
        st.number_input(
            "Sync Timeout",
            key="sync_timeout",
            min_value=1,
            help="The timeout value for the synchronization operation.",
        )

    def _set_runner_num(self):
        st.number_input("Runner Num", key="runner_num", min_value=1)

    def _set_dtype(self):
        st.selectbox("Dtype", ["float16", "bfloat16", "float32"], key="dtype")

    def _set_temperature(self):
        st.number_input("Temperature", key="temperature", min_value=0.0, max_value=2.0)

    def _set_top_p(self):
        st.number_input("Top-p", key="top_p", min_value=0.0, max_value=1.0)

    def _set_top_k(self):
        st.number_input(
            "Top-k",
            key="top_k",
            min_value=-1,
            max_value=512,
            help="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.",
        )

    def _set_seed(self):
        st.number_input("Seed", key="seed", step=1)

    def _set_logprobs(self):
        st.number_input("Logprobs", key="logprobs", min_value=0, max_value=20)

    def _set_use_v1(self):
        st.checkbox("Use V1 Engine", key="use_v1")

    def _set_enable_prefix_caching(self):
        st.checkbox("Prefix Caching", key="enable_prefix_caching")

    def _set_enforce_eager(self):
        st.checkbox("Enforce Eager", key="enforce_eager")

    def _set_gpu_memory_utilization(self):
        st.number_input(
            "GPU Memory Utilization", key="gpu_memory_utilization", min_value=0.0, max_value=1.0
        )

    def _set_enable_chunked_prefill(self):
        st.checkbox("Chunked Prefill", key="enable_chunked_prefill")

    def _set_enable_thinking(self):
        st.checkbox("Enable Thinking For Qwen3", key="enable_thinking")

    def _set_enable_openai_api(self):
        st.checkbox("Enable OpenAI API", key="enable_openai_api")

    def _set_max_timeout(self):
        st.number_input("Max Timeout", key="max_timeout", min_value=0)

    def _set_explorer_max_retry_times(self):
        st.number_input("Explorer Max Retry Times", key="explorer_max_retry_times", min_value=0)

    def _set_trainer_type(self):
        st.selectbox("Trainer Type", ["verl"], key="trainer_type")

    def _set_algorithm_type(self):
        def on_change():
            if st.session_state["algorithm_type"] == AlgorithmType.PPO.value:
                st.session_state["mode"] = "both"
                st.session_state["adv_estimator"] = AdvantageEstimator.GAE.value
            elif st.session_state["algorithm_type"] == AlgorithmType.GRPO.value:
                st.session_state["mode"] = "both"
                st.session_state["adv_estimator"] = AdvantageEstimator.GRPO.value
            elif st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
                st.session_state["mode"] = "train"
                st.session_state["adv_estimator"] = AdvantageEstimator.GRPO.value
            elif st.session_state["algorithm_type"] == AlgorithmType.OPMD.value:
                st.session_state["mode"] = "both"
                st.session_state["adv_estimator"] = AdvantageEstimator.GRPO.value
            else:  # TODO: add more algorithms
                pass
            self._set_trainer_gpu_num()

        st.selectbox(
            "Algorithm Type",
            [
                AlgorithmType.PPO.value,
                AlgorithmType.GRPO.value,
                AlgorithmType.DPO.value,
                AlgorithmType.OPMD.value,
            ],
            key="algorithm_type",
            on_change=on_change,
        )

    def _set_sft_warmup_steps(self):
        st.number_input("SFT Warmup Steps", key="sft_warmup_steps", min_value=0)

    def _set_eval_interval(self):
        st.number_input("Eval Interval", key="eval_interval", min_value=1)

    def _set_eval_on_latest_checkpoint(self):
        st.checkbox("Eval on Latest Checkpoint", key="eval_on_latest_ckp")

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

    def _set_save_interval(self):
        if (
            st.session_state["algorithm_type"] == AlgorithmType.DPO.value
            or st.session_state["sync_method"] == SyncMethod.NCCL.value
        ):
            st.session_state["save_interval"] = st.session_state["_nccl_save_interval"]
            freeze_save_interval = False
        else:
            st.session_state["save_interval"] = st.session_state["sync_interval"]
            freeze_save_interval = True

        def on_change():
            if (
                st.session_state["algorithm_type"] == AlgorithmType.DPO.value
                or st.session_state["sync_method"] == SyncMethod.NCCL.value
            ):
                st.session_state["_nccl_save_interval"] = st.session_state["save_interval"]

        st.number_input(
            "Save Interval",
            key="save_interval",
            min_value=1,
            help="Set to `sync_interval` when `algorithm_type != DPO && sync_method == checkpoint`",
            disabled=freeze_save_interval,
            on_change=on_change,
        )

    def _set_ppo_epochs(self):
        st.number_input("PPO Epochs", key="ppo_epochs", min_value=1)

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
        st.number_input("Critic Warmup Steps", key="critic_warmup", min_value=0)

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
        st.number_input(r"Gamma :blue-badge[$\gamma$]", key="gamma")

    def _set_lam(self):
        st.number_input(r"Lambda :blue-badge[$\lambda$]", key="lam")

    def _set_norm_adv_by_std_in_grpo(self):
        st.checkbox("Norm Adv by Std in GRPO", key="norm_adv_by_std_in_grpo")

    def _set_use_kl_in_reward(self):
        st.checkbox("Use KL in Reward", key="use_kl_in_reward")

    def _set_kl_penalty(self):
        st.selectbox("KL Penalty", ["kl", "abs", "mse", "low_var_kl"], key="kl_penalty")

    def _set_kl_ctrl_type(self):
        st.selectbox("KL Ctrl Type", ["fixed", "adaptive"], key="kl_ctrl_type")

    def _set_kl_ctrl_coef(self):
        st.number_input("KL Ctrl Coef", key="kl_ctrl_coef", format="%.1e")

    def _set_horizon(self):
        st.number_input("Horizon", key="horizon", min_value=1.0)

    def _set_target_kl(self):
        st.number_input("Target KL", key="target_kl", format="%.1e")

    def _set_actor_ppo_micro_batch_size_per_gpu(self):
        st.session_state["actor_ppo_micro_batch_size_per_gpu"] = min(
            st.session_state["actor_ppo_micro_batch_size_per_gpu"],
            st.session_state["_train_batch_size_per_gpu"],
        )
        st.number_input(
            "Micro Batch Size Per GPU :blue-badge[(Actor)]",
            key="actor_ppo_micro_batch_size_per_gpu",
            min_value=1,
            max_value=st.session_state["_train_batch_size_per_gpu"],
        )

    def _set_ref_log_prob_micro_batch_size_per_gpu(self):
        st.session_state["ref_log_prob_micro_batch_size_per_gpu"] = min(
            st.session_state["ref_log_prob_micro_batch_size_per_gpu"],
            st.session_state["_train_batch_size_per_gpu"],
        )
        st.number_input(
            "Micro Batch Size Per GPU :blue-badge[(Ref)]",
            key="ref_log_prob_micro_batch_size_per_gpu",
            min_value=1,
            max_value=st.session_state["_train_batch_size_per_gpu"],
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
            "Learning Rate :blue-badge[(Actor)]",
            key="actor_lr",
            min_value=1e-7,
            max_value=1e-3,
            format="%.1e",
        )

    def _set_actor_warmup_style(self):
        st.selectbox(
            "LR Warmup Style :blue-badge[(Actor)]",
            ["constant", "cosine"],
            key="actor_warmup_style",
        )

    def _set_actor_lr_warmup_steps_ratio(self):
        st.number_input(
            "LR Warmup Steps Ratio :blue-badge[(Actor)]",
            key="actor_lr_warmup_steps_ratio",
            min_value=0.0,
            max_value=1.0,
        )

    def _set_actor_grad_clip(self):
        st.number_input(
            "Grad Clip :blue-badge[(Actor)]",
            key="actor_grad_clip",
            min_value=0.0,
            max_value=1.0,
            help="Clipping by Norm",
        )

    def _set_actor_clip_ratio(self):
        st.number_input(
            r"Clip Ratio :blue-badge[$\epsilon$]",
            key="actor_clip_ratio",
            min_value=0.0,
            max_value=1.0,
        )

    def _set_actor_entropy_coef(self):
        st.number_input(
            "Entropy Coeff",
            key="actor_entropy_coef",
            min_value=0.0,
            max_value=1.0,
            format="%.1e",
        )

    def _set_actor_use_kl_loss(self):
        if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
            st.session_state["actor_use_kl_loss"] = True
        else:
            st.session_state["actor_use_kl_loss"] = st.session_state["_not_dpo_actor_use_kl_loss"]

            def on_change():
                st.session_state["_not_dpo_actor_use_kl_loss"] = st.session_state[
                    "actor_use_kl_loss"
                ]

            st.checkbox("Use KL Loss", key="actor_use_kl_loss", on_change=on_change)

    def _set_actor_kl_loss_coef(self):
        st.number_input(
            r"KL Loss Coef :blue-badge[$\beta$]",
            key="actor_kl_loss_coef",
            min_value=0.0,
            max_value=1.0,
            format="%.1e",
        )

    def _set_actor_kl_loss_type(self):
        st.selectbox(
            "KL Loss Type",
            ["kl", "abs", "mse", "low_var_kl"],
            key="actor_kl_loss_type",
        )

    def _set_actor_tau(self):
        st.number_input(
            "Tau for OPMD",
            key="actor_tau",
            min_value=0.0,
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
        st.session_state["critic_ppo_micro_batch_size_per_gpu"] = min(
            st.session_state["critic_ppo_micro_batch_size_per_gpu"],
            st.session_state["_train_batch_size_per_gpu"],
        )
        st.number_input(
            "Micro Batch Size Per GPU :blue-badge[(Critic)]",
            key="critic_ppo_micro_batch_size_per_gpu",
            min_value=1,
            max_value=st.session_state["_train_batch_size_per_gpu"],
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
            "Learning Rate :blue-badge[(Critic)]",
            key="critic_lr",
            min_value=1e-7,
            max_value=1e-3,
            format="%.1e",
        )

    def _set_critic_warmup_style(self):
        st.selectbox(
            "LR Warmup Style :blue-badge[(Critic)]",
            ["constant", "cosine"],
            key="critic_warmup_style",
        )

    def _set_critic_lr_warmup_steps_ratio(self):
        st.number_input(
            "LR Warmup Steps Ratio :blue-badge[(Critic)]",
            key="critic_lr_warmup_steps_ratio",
            min_value=0.0,
            max_value=1.0,
        )

    def _set_critic_grad_clip(self):
        st.number_input(
            "Grad Clip :blue-badge[(Critic)]",
            key="critic_grad_clip",
            min_value=0.0,
            max_value=1.0,
            help="Clipping by Norm",
        )

    def _set_critic_cliprange_value(self):
        st.number_input(
            "Cliprange Value",
            key="critic_cliprange_value",
            min_value=0.0,
            max_value=1.0,
        )

    def _set_critic_checkpoint(self):
        st.multiselect(
            "Checkpoint",
            ["model", "hf_model", "optimizer", "extra"],
            key="critic_checkpoint",
        )

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
        self._set_configs_with_st_columns(["project", "exp_name"], columns_config=[1, 3])

        self._set_model_path()

        self._set_checkpoint_root_dir()

        self._set_taskset_path()

        self._set_configs_with_st_columns(["algorithm_type", "sft_warmup_steps", "monitor_type"])
        if st.session_state["sft_warmup_steps"] > 0:
            self._set_sft_warmup_dataset_path()

        st.header("Important Configs")
        self._set_configs_with_st_columns(
            ["node_num", "gpu_per_node", "engine_num", "tensor_parallel_size"]
            if st.session_state["mode"] == "both"
            else ["node_num", "gpu_per_node"]
        )
        self._check_engine_num_and_tp_size()

        self._set_configs_with_st_columns(
            ["total_epochs", "train_batch_size", "ppo_epochs", "repeat_times"]
            if st.session_state["mode"] == "both"
            else ["total_epochs", "train_batch_size", "ppo_epochs"]
        )
        self._check_train_batch_size()

        self._set_configs_with_st_columns(["max_prompt_tokens", "max_response_tokens"])

        self._set_configs_with_st_columns(
            ["sync_interval", "eval_interval", "save_interval"]
            if st.session_state["mode"] == "both"
            else ["eval_interval", "save_interval"]
        )

        if st.session_state["algorithm_type"] != AlgorithmType.DPO.value:
            self._set_taskset_args()
        else:
            self._set_dpo_dataset_kwargs()

        if st.session_state["sft_warmup_steps"] > 0:
            self._set_sft_warmup_dataset_args()

        self._set_configs_with_st_columns(["default_workflow_type", "default_reward_fn_type"])

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
            st.session_state["adv_estimator"] == AdvantageEstimator.GAE.value
        )  # TODO: may apply to expert mode
        if use_critic:
            self._set_configs_with_st_columns(["critic_ppo_micro_batch_size_per_gpu", "critic_lr"])

    def _expert_model_part(self):
        self._set_configs_with_st_columns(["project", "exp_name"], columns_config=[1, 3])

        self._set_model_path()
        self._set_critic_model_path()

        self._set_checkpoint_root_dir()

        self._set_configs_with_st_columns(["monitor_type", "node_num", "gpu_per_node"])
        self._set_configs_with_st_columns(["max_prompt_tokens", "max_response_tokens"])

    def _expert_buffer_part(self):
        self._set_configs_with_st_columns(["total_epochs", "train_batch_size"])
        self._check_train_batch_size()

        self._set_configs_with_st_columns(["default_workflow_type", "default_reward_fn_type"])
        self._set_system_prompt()
        self._set_reply_prefix()

        if st.session_state["algorithm_type"] != AlgorithmType.DPO.value:
            with st.expander("Taskset Configs", expanded=True):
                self._set_taskset_path()
                self._set_taskset_args()
        else:
            with st.expander("DPO Dataset Configs", expanded=True):
                self._set_experience_buffer_path()
                self._set_dpo_dataset_kwargs()

        with st.expander("Eval Tasksets Configs", expanded=True):
            self._set_eval_tasksets()

        with st.expander("SFT Dataset Configs"):
            self._set_sft_warmup_dataset_path()
            self._set_sft_warmup_dataset_args()

        if st.session_state["algorithm_type"] != AlgorithmType.DPO.value:
            with st.expander("Experiences Buffer Configs", expanded=True):
                self._set_storage_type()
                self._set_experience_buffer_path()

        self.buffer_advanced_tab = st.expander("Advanced Config")
        with self.buffer_advanced_tab:
            self._set_configs_with_st_columns(["buffer_max_retry_times", "max_retry_interval"])

    def _expert_explorer_part(self):
        self._set_configs_with_st_columns(["sync_method", "sync_interval", "sync_timeout"])

        self._set_configs_with_st_columns(
            [
                "runner_num",
                "max_timeout",
                "explorer_max_retry_times",
            ]
        )

        self._set_configs_with_st_columns(["eval_interval", "eval_on_latest_checkpoint"])

        with st.expander("Rollout Model Config", expanded=True):
            self._set_configs_with_st_columns(["engine_type", "engine_num", "tensor_parallel_size"])
            self._check_engine_num_and_tp_size()

            self._set_configs_with_st_columns(["gpu_memory_utilization", "dtype", "seed"])

            self._set_configs_with_st_columns(
                ["use_v1", "enforce_eager", "enable_prefix_caching", "enable_chunked_prefill"]
            )

            self._set_configs_with_st_columns(["enable_thinking", "enable_openai_api"])

        with st.expander("Auxiliary Models", expanded=True):  # TODO
            pass

    def _expert_trainer_part(self):
        self._set_configs_with_st_columns(["algorithm_type", "gamma", "lam"])
        self._set_configs_with_st_columns(["repeat_times", "save_interval"])
        self._check_sft_warmup_dataset_path()

        if st.session_state["trainer_type"] == "verl":
            self._expert_verl_trainer_part()

    def _expert_verl_trainer_part(self):
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

            self._set_configs_with_st_columns(["ppo_epochs", "training_strategy", "resume_mode"])

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
            self._set_configs_with_st_columns(["norm_adv_by_std_in_grpo", "use_kl_in_reward"])
            self._set_configs_with_st_columns(["kl_penalty", "kl_ctrl_type", "kl_ctrl_coef"])
            self._set_configs_with_st_columns(["horizon", "target_kl"])

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
                ["actor_grad_clip", "actor_clip_ratio", "actor_entropy_coef"]
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
            self._set_critic_checkpoint()

    def expert_mode(self):
        tab2func = {
            "Model": self._expert_model_part,
            "Buffer": self._expert_buffer_part,
            "Explorer and Synchronizer": self._expert_explorer_part,
            "Trainer": self._expert_trainer_part,
        }
        if st.session_state["mode"] == "train":
            del tab2func["Explorer and Synchronizer"]
        tabs = st.tabs(list(tab2func.keys()))
        for tab, func in zip(tabs, tab2func.values()):
            with tab:
                func()

    def _generate_verl_config(self):
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

        ppo_max_token_len_per_gpu = st.session_state["repeat_times"] * (
            st.session_state["max_prompt_tokens"] + st.session_state["max_response_tokens"]
        )

        trainer_config = {
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "model": {
                    "external_lib": None,
                    "override_config": {},
                    "enable_gradient_checkpointing": enable_gradient_checkpointing,
                    "use_remove_padding": use_remove_padding,
                },
                "actor": {
                    "strategy": st.session_state["training_strategy"],
                    "ppo_mini_batch_size": st.session_state["train_batch_size"],
                    "ppo_micro_batch_size_per_gpu": st.session_state[
                        "actor_ppo_micro_batch_size_per_gpu"
                    ],
                    "use_dynamic_bsz": use_dynamic_bsz,
                    "ppo_max_token_len_per_gpu": ppo_max_token_len_per_gpu,
                    "kl_loss_type": st.session_state["actor_kl_loss_type"],
                    "ppo_epochs": st.session_state["ppo_epochs"],
                    "shuffle": False,
                    "ulysses_sequence_parallel_size": st.session_state[
                        "actor_ulysses_sequence_parallel_size"
                    ],
                    "checkpoint": {"contents": st.session_state["actor_checkpoint"]},
                    "optim": {
                        "lr": st.session_state["actor_lr"],
                        "lr_warmup_steps_ratio": st.session_state["actor_lr_warmup_steps_ratio"],
                        "warmup_style": st.session_state["actor_warmup_style"],
                        "total_training_steps": (
                            -1
                            if st.session_state["total_training_steps"] is None
                            else st.session_state["total_training_steps"]
                        ),
                    },
                    "fsdp_config": copy.deepcopy(fsdp_config),
                    "tau": st.session_state["actor_tau"],
                    "opmd_baseline": st.session_state["actor_opmd_baseline"],
                    "use_uid": st.session_state["actor_use_uid"],
                },
                "ref": {
                    "fsdp_config": copy.deepcopy(fsdp_config),
                    "log_prob_micro_batch_size_per_gpu": st.session_state[
                        "ref_log_prob_micro_batch_size_per_gpu"
                    ],
                    "log_prob_use_dynamic_bsz": use_dynamic_bsz,
                    "log_prob_max_token_len_per_gpu": ppo_max_token_len_per_gpu,
                    "ulysses_sequence_parallel_size": st.session_state[
                        "actor_ulysses_sequence_parallel_size"
                    ],
                },
            },
            "custom_reward_function": {"path": None, "name": "compute_score"},
            "algorithm": {
                "kl_penalty": st.session_state["kl_penalty"],
                "kl_ctrl": {
                    "type": st.session_state["kl_ctrl_type"],
                    "kl_coef": st.session_state["kl_ctrl_coef"],
                },
            },
            "trainer": {
                "balance_batch": balance_batch,
                "logger": ["tensorboard"],
                "resume_mode": st.session_state["resume_mode"],
                "resume_from_path": st.session_state["resume_from_path"],
                "default_hdfs_dir": st.session_state["default_hdfs_dir"],
                "remove_previous_ckpt_in_save": st.session_state["remove_previous_ckpt_in_save"],
                "del_local_ckpt_after_load": st.session_state["del_local_ckpt_after_load"],
                "val_before_train": False,
                "max_actor_ckpt_to_keep": st.session_state["max_actor_ckpt_to_keep"],
                "max_critic_ckpt_to_keep": st.session_state["max_critic_ckpt_to_keep"],
            },
        }

        if st.session_state["adv_estimator"] == AdvantageEstimator.GAE.value:
            trainer_config["trainer"]["critic_warmup"] = st.session_state["critic_warmup"]
            trainer_config["critic"] = {
                "strategy": st.session_state["training_strategy"],
                "optim": {
                    "lr": st.session_state["critic_lr"],
                    "lr_warmup_steps_ratio": st.session_state["critic_lr_warmup_steps_ratio"],
                    "warmup_style": st.session_state["critic_warmup_style"],
                    "total_training_steps": (
                        -1
                        if st.session_state["total_training_steps"] is None
                        else st.session_state["total_training_steps"]
                    ),
                },
                "model": {
                    "override_config": {},
                    "external_lib": None,
                    "enable_gradient_checkpointing": enable_gradient_checkpointing,
                    "use_remove_padding": use_remove_padding,
                    "fsdp_config": copy.deepcopy(fsdp_config),
                },
                "ppo_mini_batch_size": st.session_state["train_batch_size"],
                "ppo_micro_batch_size_per_gpu": st.session_state[
                    "critic_ppo_micro_batch_size_per_gpu"
                ],
                "forward_micro_batch_size_per_gpu": st.session_state[
                    "critic_ppo_micro_batch_size_per_gpu"
                ],
                "use_dynamic_bsz": use_dynamic_bsz,
                "ppo_max_token_len_per_gpu": ppo_max_token_len_per_gpu * 2,
                "forward_max_token_len_per_gpu": ppo_max_token_len_per_gpu * 2,
                "ulysses_sequence_parallel_size": st.session_state[
                    "critic_ulysses_sequence_parallel_size"
                ],
                "ppo_epochs": st.session_state["ppo_epochs"],
                "shuffle": False,
                "grad_clip": st.session_state["critic_grad_clip"],
                "cliprange_value": st.session_state["critic_cliprange_value"],
                "checkpoint": {"contents": st.session_state["critic_checkpoint"]},
            }
        return trainer_config

    def _gen_buffer_config(self):
        if st.session_state["algorithm_type"] != AlgorithmType.DPO.value:
            experience_buffer_path = st.session_state["experience_buffer_path"].strip()
            if (
                not experience_buffer_path
                and st.session_state["storage_type"] == StorageType.SQL.value
            ):
                experience_buffer_path = f"sqlite:///{os.path.join(st.session_state['checkpoint_root_dir'], '.cache', st.session_state['project'], st.session_state['exp_name'])}/data.db"

        sft_storage_type = (
            StorageType.SQL.value
            if "://" in st.session_state["sft_warmup_dataset_path"]
            else StorageType.FILE.value
        )  # TODO

        buffer_config = {
            "batch_size": st.session_state["train_batch_size"],
            "total_epochs": st.session_state["total_epochs"],
            "explorer_input": {
                "taskset": {
                    "name": "taskset",
                    "storage_type": StorageType.FILE.value,
                    "path": st.session_state["taskset_path"],
                    "split": st.session_state["taskset_split"],
                    "subset_name": st.session_state["taskset_subset_name"],
                    "format": {
                        "prompt_key": st.session_state["taskset_prompt_key"],
                        "response_key": st.session_state["taskset_response_key"],
                    },
                    "rollout_args": {
                        "temperature": st.session_state["temperature"],
                        "logprobs": st.session_state["logprobs"],
                    },
                },
                "eval_tasksets": [],
                "default_workflow_type": st.session_state["default_workflow_type"],
                "default_reward_fn_type": st.session_state["default_reward_fn_type"],
                "system_prompt": st.session_state["system_prompt"],
                "reply_prefix": st.session_state["reply_prefix"],
            },
            "trainer_input": {
                "experience_buffer": {
                    "name": "experience_buffer",
                    "storage_type": st.session_state["storage_type"],
                    "path": experience_buffer_path,
                },
                "sft_warmup_steps": st.session_state["sft_warmup_steps"],
            },
            "max_retry_times": st.session_state["buffer_max_retry_times"],
            "max_retry_interval": st.session_state["max_retry_interval"],
        }

        for idx in range(st.session_state["_eval_tasksets_num"]):
            if st.session_state[f"eval_taskset_{idx}_path"].strip():
                buffer_config["explorer_input"]["eval_tasksets"].append(
                    {
                        "name": st.session_state[f"eval_taskset_{idx}_name"],
                        "path": st.session_state[f"eval_taskset_{idx}_path"],
                        "subset_name": st.session_state[f"eval_taskset_{idx}_subset_name"],
                        "split": st.session_state[f"eval_taskset_{idx}_split"],
                        "prompt_key": st.session_state[f"eval_taskset_{idx}_prompt_key"],
                        "response_key": st.session_state[f"eval_taskset_{idx}_response_key"],
                    }
                )
        if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
            experience_buffer = buffer_config["trainer_input"]["experience_buffer"]
            experience_buffer["split"] = st.session_state["dpo_dataset_train_split"]
            experience_buffer["format"] = {
                "prompt_type": st.session_state["dpo_dataset_prompt_type"],
                "prompt_key": st.session_state["dpo_dataset_prompt_key"],
                "chosen_key": st.session_state["dpo_dataset_chosen_key"],
                "rejected_key": st.session_state["dpo_dataset_rejected_key"],
            }
        if st.session_state["sft_warmup_dataset_path"].strip():
            buffer_config["trainer_input"]["sft_warmup_dataset"] = {
                "name": "sft_warmup_dataset",
                "storage_type": sft_storage_type,
                "path": st.session_state["sft_warmup_dataset_path"],
                "split": st.session_state["sft_warmup_train_split"],
                "format": {
                    "prompt_type": st.session_state["sft_warmup_prompt_type"],
                    "messages_key": st.session_state["sft_warmup_messages_key"],
                    "prompt_key": st.session_state["sft_warmup_prompt_key"],
                    "response_key": st.session_state["sft_warmup_response_key"],
                },
            }

        return buffer_config

    def _gen_explorer_config(self):
        explorer_config = {
            "runner_num": st.session_state["runner_num"],
            "max_timeout": st.session_state["max_timeout"],
            "max_retry_times": st.session_state["explorer_max_retry_times"],
            "rollout_model": {
                "engine_type": st.session_state["engine_type"],
                "engine_num": st.session_state["engine_num"],
                "tensor_parallel_size": st.session_state["tensor_parallel_size"],
                "use_v1": st.session_state["use_v1"],
                "enforce_eager": st.session_state["enforce_eager"],
                "enable_prefix_caching": st.session_state["enable_prefix_caching"],
                "enable_chunked_prefill": st.session_state["enable_chunked_prefill"],
                "gpu_memory_utilization": st.session_state["gpu_memory_utilization"],
                "dtype": st.session_state["dtype"],
                "seed": st.session_state["seed"],
                # "max_prompt_tokens": None,  # TODO
                # "max_response_tokens": None,  # TODO
                # "chat_template": None,  # TODO: add chat template
                "enable_thinking": st.session_state["enable_thinking"],
                "enable_openai_api": st.session_state["enable_openai_api"],
            },
            "auxiliary_models": [],
            "eval_interval": st.session_state["eval_interval"],
            "eval_on_latest_checkpoint": st.session_state["eval_on_latest_checkpoint"],
        }
        return explorer_config

    def generate_config(self):
        if st.session_state["trainer_type"] == "verl":
            trainer_config = self._generate_verl_config()
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
            use_container_width=True,
            icon=":material/create_new_folder:",
        ):
            st.session_state.config_generated = True
            st.session_state.is_running = False
        if st.session_state.config_generated:
            config = {
                "mode": st.session_state["mode"],
                "project": st.session_state["project"],
                "name": st.session_state["exp_name"],
                "checkpoint_root_dir": st.session_state["checkpoint_root_dir"],
                "algorithm": {
                    "algorithm_type": st.session_state["algorithm_type"],
                    "repeat_times": st.session_state["repeat_times"],
                    "gamma": st.session_state["gamma"],
                    "lam": st.session_state["lam"],
                },
                "data_processor": {},  # TODO: Add data processor config
                "model": {
                    "model_path": st.session_state["model_path"],
                    "max_prompt_tokens": st.session_state["max_prompt_tokens"],
                    "max_response_tokens": st.session_state["max_response_tokens"],
                },
                "cluster": {
                    "node_num": st.session_state["node_num"],
                    "gpu_per_node": st.session_state["gpu_per_node"],
                },
                "buffer": self._gen_buffer_config(),
                "explorer": self._gen_explorer_config(),
                "trainer": {
                    "trainer_type": st.session_state["trainer_type"],
                    "save_interval": st.session_state["save_interval"],
                    "enable_preview": True,  # TODO
                    "actor_use_kl_loss": st.session_state["actor_use_kl_loss"],
                    "actor_kl_loss_coef": st.session_state["actor_kl_loss_coef"],
                    "actor_entropy_coef": st.session_state["actor_entropy_coef"],
                    "actor_grad_clip": st.session_state["actor_grad_clip"],
                    "actor_clip_ratio": st.session_state["actor_clip_ratio"],
                    "trainer_config": trainer_config,
                },
                "monitor": {
                    "monitor_type": st.session_state["monitor_type"],
                },
                "synchronizer": {
                    "sync_method": st.session_state["sync_method"],
                    "sync_interval": st.session_state["sync_interval"],
                    "sync_timeout": st.session_state["sync_timeout"],
                },
            }

            if st.session_state["adv_estimator"] == AdvantageEstimator.GAE.value:
                config["model"]["critic_model_path"] = (
                    st.session_state["critic_model_path"].strip()
                    if st.session_state["critic_model_path"].strip()
                    else st.session_state["model_path"]
                )

            st.session_state.config_generated = True
            st.header("Generated Config File")
            buttons = st.container()
            save_btn, run_btn = buttons.columns(2, vertical_alignment="bottom")
            yaml_config = yaml.dump(config, allow_unicode=True, sort_keys=False)
            save_btn.download_button(
                "Save",
                data=yaml_config,
                file_name=f"{config['project']}-{config['name']}.yaml",
                mime="text/plain",
                icon=":material/download:",
                use_container_width=True,
            )
            run_btn.button(
                "Run",
                on_click=self.run_config,
                args=(
                    buttons,
                    yaml_config,
                ),
                icon=":material/terminal:",
                use_container_width=True,
                disabled=st.session_state.is_running,
            )
            st.code(yaml_config, language="yaml")

    def run_config(self, parent, yaml_config: str) -> None:
        st.session_state.is_running = True

        import ray

        # first check if ray is running
        ray_status = subprocess.run(
            ["ray", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if ray_status.returncode != 0:
            parent.warning(
                "Ray cluster is not running. Please start Ray first using `ray start --head`."
            )
            return
        context = ray.init(ignore_reinit_error=True)
        dashboard_url = context.dashboard_url
        # save config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmpfile:
            tmpfile.write(yaml_config)
            tmpfile_path = tmpfile.name

        # submit ray job
        try:
            subprocess.run(
                [
                    "ray",
                    "job",
                    "submit",
                    "--no-wait",
                    "--",
                    "python",
                    "-m",
                    "trinity.cli.launcher",
                    "run",
                    "--config",
                    tmpfile_path,
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            parent.success(
                f"Job submitted successfully!\n\n"
                f"View progress in the Ray Dashboard: http://{dashboard_url}",
                icon="✅",
            )
        except subprocess.CalledProcessError as e:
            parent.error(f"Failed to submit job:\n\n{e.stderr}", icon="❌")
            st.session_state.is_running = False


if __name__ == "__main__":
    config_manager = ConfigManager()
