import os

import streamlit as st
import yaml
from verl.trainer.ppo.ray_trainer import AdvantageEstimator

from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows.workflow import WORKFLOWS


def main():  # noqa: C901
    st.title("Config File Generator")
    unfinished_flag = False

    model_tab, buffer_tab, connector_tab, trainer_tab = st.tabs(
        ["Model", "Buffer", "Explorer and Synchronizer", "Trainer"]
    )
    with model_tab:
        project_col, name_col = st.columns([1, 3])
        project = project_col.text_input("Project", "Trinity-RFT")
        name = name_col.text_input("Experiment Name", "qwen2.5-1.5B")

        model_path = st.text_input("Model Path", "")
        if not model_path.strip():
            unfinished_flag = True
            st.warning("Please input model path")
        critic_model_path = st.text_input("Critic Model Path (defaults to `model_path`)", "")
        if not critic_model_path.strip():
            critic_model_path = model_path

        checkpoint_path = st.text_input("Checkpoint Path", "")
        if not checkpoint_path.strip():
            unfinished_flag = True
            st.warning("Please input checkpoint path")

        node_num_col, gpu_per_node_col, max_prompt_tokens_col, max_response_tokens_col = st.columns(
            4
        )
        node_num = node_num_col.number_input("Node Num", value=1, min_value=1)
        gpu_per_node = gpu_per_node_col.number_input(
            "GPU Per Node", value=8, min_value=1, max_value=8
        )
        max_prompt_tokens = max_prompt_tokens_col.number_input(
            "Max Prompt Tokens", value=256, min_value=1
        )
        max_response_tokens = max_response_tokens_col.number_input(
            "Max Response Tokens", value=1024, min_value=1
        )

    with buffer_tab:
        total_epoch_col, batch_size_per_gpu_col = st.columns(2)
        total_epoch = total_epoch_col.number_input("Total Epoch", value=20, min_value=1)
        batch_size_per_gpu = batch_size_per_gpu_col.number_input(
            "Batch Size Per GPU", value=1, min_value=1
        )

        dataset_path = st.text_input("Dataset Path", "")
        if not dataset_path.strip():
            unfinished_flag = True
            st.warning("Please input dataset path")

        if dataset_path and "://" not in dataset_path:
            train_split_col, eval_split_col, prompt_key_col, response_key_col = st.columns(4)
            train_split = train_split_col.text_input("Train Split", "train")
            eval_split = eval_split_col.text_input("Eval Split", "")
            prompt_key = prompt_key_col.text_input("Prompt Key", "question")
            response_key = response_key_col.text_input("Response Key", "answer")

        default_workflow_type_col, default_reward_fn_type_col, storage_type_col = st.columns(3)
        default_workflow_type = default_workflow_type_col.selectbox(
            "Default Workflow Type", WORKFLOWS.modules.keys(), index=1
        )
        default_reward_fn_type = default_reward_fn_type_col.selectbox(
            "Default Reward Fn Type", REWARD_FUNCTIONS.modules.keys(), index=3
        )
        storage_type = storage_type_col.selectbox(
            "Storage Type", ["sql", "redis", "queue"], index=2
        )

        buffer_advanced_tab = st.expander("Advanced Config")
        with buffer_advanced_tab:
            db_url = st.text_input(
                "DB URL",
                "",
                help=r"Default to `sqlite:///{os.path.join(checkpoint_path, '.cache', project, name)}/data.db`",
            )
            if not db_url.strip():
                db_url = (
                    rf"sqlite:///{os.path.join(checkpoint_path, '.cache', project, name)}/data.db"
                )

            max_retry_times_col, max_retry_interval_col = st.columns(2)
            max_retry_times = max_retry_times_col.number_input(
                "Max Retry Times", value=3, min_value=1
            )
            max_retry_interval = max_retry_interval_col.number_input(
                "Max Retry Interval", value=1, min_value=1
            )

            sft_warmup_dataset_path = st.text_input("SFT Warmup Dataset Path", "")
            if sft_warmup_dataset_path and "://" not in sft_warmup_dataset_path:  # TODO
                (
                    sft_warmup_train_split_col,
                    sft_warmup_eval_split_col,
                    sft_warmup_prompt_key_col,
                    sft_warmup_response_key_col,
                ) = st.columns(4)
                sft_warmup_train_split = sft_warmup_train_split_col.text_input(  # noqa: F841
                    "SFT Train Split", "train"
                )
                sft_warmup_eval_split = sft_warmup_eval_split_col.text_input(  # noqa: F841
                    "SFT Eval Split", ""
                )
                sft_warmup_prompt_key = sft_warmup_prompt_key_col.text_input(  # noqa: F841
                    "SFT Prompt Key", "question"
                )
                sft_warmup_response_key = sft_warmup_response_key_col.text_input(  # noqa: F841
                    "SFT Response Key", "answer"
                )
            else:
                sft_warmup_train_split = ""  # noqa: F841
                sft_warmup_eval_split = ""  # noqa: F841
                sft_warmup_prompt_key = ""  # noqa: F841
                sft_warmup_response_key = ""  # noqa: F841

    with connector_tab:
        engine_type_col, engine_num_col, tensor_parallel_size_col, repeat_times_col = st.columns(4)
        engine_type = engine_type_col.selectbox(
            "Explorer Engine Type", ["vllm_async", "vllm"], index=0
        )
        if "engine_num" not in st.session_state:
            st.session_state.engine_num = 2
        old_engine_num = min(st.session_state.engine_num, gpu_per_node * node_num)
        engine_num = engine_num_col.number_input(
            "Engine Num",
            value=old_engine_num,
            min_value=1,
            max_value=gpu_per_node * node_num,
            help="cannot exceed `gpu_per_node` * `node_num`",
        )
        st.session_state.engine_num = engine_num
        tensor_parallel_size = tensor_parallel_size_col.number_input(
            "Tensor Parallel Size", value=1, min_value=1, max_value=8
        )
        repeat_times = repeat_times_col.number_input("Repeat Times", value=1, min_value=1)

        sync_method_col, sync_iteration_interval_col = st.columns(2)
        sync_method = sync_method_col.selectbox("Sync Method", ["online", "offline"], index=0)
        sync_iteration_interval = sync_iteration_interval_col.number_input(
            "Sync Iteration Interval", value=10, min_value=1
        )
        with st.expander("Advanced Config"):
            runner_num_col, max_pending_requests_col, max_waiting_steps_col, dtype_col = st.columns(
                4
            )
            runner_num = runner_num_col.number_input("Runner Num", value=32, min_value=1)
            max_pending_requests = max_pending_requests_col.number_input(
                "Max Pending Requests", value=32, min_value=1
            )
            max_waiting_steps = max_waiting_steps_col.number_input(
                "Max Waiting Steps", value=4, min_value=1
            )
            dtype = dtype_col.selectbox("Dtype", ["float16", "bfloat16", "float32"], index=1)

            backend_col, temperature_col, top_p_col, top_k_col, seed_col, logprobs_col = st.columns(
                6
            )
            backend = backend_col.selectbox("Backend", ["nccl"], index=0)
            temperature = temperature_col.number_input(
                "Temperature", value=1.0, min_value=0.0, max_value=2.0
            )
            top_p = top_p_col.number_input("Top P", value=1.0, min_value=0.0, max_value=1.0)
            top_k = top_k_col.number_input("Top K", value=1, min_value=1, max_value=512)
            seed = seed_col.number_input("Seed", value=42)
            logprobs = logprobs_col.number_input("Logprobs", value=0, min_value=0, max_value=20)

            enable_prefix_caching_col, enforce_eager_col = st.columns(2)
            enable_prefix_caching = enable_prefix_caching_col.checkbox(
                "Enable Prefix Caching", value=False
            )
            enforce_eager = enforce_eager_col.checkbox("Enforce Eager", value=True)

    gpu_num = gpu_per_node * node_num - engine_num

    with trainer_tab:
        trainer_type_col, sft_warmup_iteration_col, eval_interval_col = st.columns(3)
        trainer_type = trainer_type_col.selectbox("Trainer Type", ["verl"], index=0)
        sft_warmup_iteration = sft_warmup_iteration_col.number_input(
            "SFT Warmup Iteration", value=0, min_value=0
        )
        if sft_warmup_iteration and not sft_warmup_dataset_path.strip():
            unfinished_flag = True
            st.warning("Please input SFT warmup dataset path when `sft_warmup_iteration` is not 0")
            with buffer_advanced_tab:
                st.warning(
                    "Please input SFT warmup dataset path when `sft_warmup_iteration` is not 0"
                )
        eval_interval = eval_interval_col.number_input("Eval Interval", value=1000, min_value=1)
        if trainer_type == "verl":
            trainer_config_path = st.text_input("Trainer Config Path", "")
            if not trainer_config_path.strip():
                unfinished_flag = True
                st.warning("Please input trainer config path")

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
                training_args = st.multiselect(
                    "Training Args",
                    ["balance_batch", "gradient_checkpointing", "remove_padding", "dynamic_bsz"],
                    default=[
                        "balance_batch",
                        "gradient_checkpointing",
                        "remove_padding",
                        "dynamic_bsz",
                    ],
                )
                balance_batch = "balance_batch" in training_args
                enable_gradient_checkpointing = "gradient_checkpointing" in training_args
                use_remove_padding = "remove_padding" in training_args
                use_dynamic_bsz = "dynamic_bsz" in training_args

                (
                    save_freq_col,
                    training_strategy_col,
                    resume_mode_col,
                    critic_warmup_col,
                ) = st.columns(4)
                if "save_freq" not in st.session_state:
                    st.session_state.save_freq = 100
                if sync_method == "online":
                    save_freq = save_freq_col.number_input(
                        "Save Freq",
                        value=st.session_state.save_freq,
                        min_value=1,
                        help="Set to `sync_iteration_interval` when `sync_method` is `offline`",
                    )
                    st.session_state.save_freq = save_freq
                else:
                    st.session_state.save_freq = sync_iteration_interval
                    save_freq = save_freq_col.number_input(
                        "Save Freq",
                        value=st.session_state.save_freq,
                        min_value=1,
                        help="Set to `sync_iteration_interval` when `sync_method` is `offline`",
                        disabled=True,
                    )

                training_strategy = training_strategy_col.selectbox(
                    "Training Strategy",
                    ["fsdp", "megatron"],
                    index=0,
                    help="megatron is not tested",
                )
                if training_strategy == "fsdp":
                    param_offload_col, optimizer_offload_col = st.columns(2)
                    param_offload = param_offload_col.checkbox("FSDP Param Offload", value=False)
                    optimizer_offload = optimizer_offload_col.checkbox(
                        "FSDP Optimizer Offload", value=False
                    )
                    fsdp_config = {
                        "wrap_policy": {"min_num_params": 0},
                        "param_offload": param_offload,
                        "optimizer_offload": optimizer_offload,
                        "fsdp_size": -1,
                    }
                else:
                    fsdp_config = {}

                resume_mode = resume_mode_col.selectbox(
                    "Resume Mode", ["disable", "auto", "resume_path"], index=1
                )
                if "resume_from_path" not in st.session_state:
                    st.session_state.resume_from_path = ""
                if resume_mode == "resume_path":
                    resume_from_path = st.text_input(
                        "Resume Path", st.session_state.resume_from_path
                    )
                    st.session_state.resume_from_path = resume_from_path
                    if not resume_from_path.strip() or "global_step_" not in resume_from_path:
                        unfinished_flag = True
                        st.warning(
                            "Please input a valid resume path when `resume_mode` is `resume_path`"
                        )
                else:
                    resume_from_path = st.session_state.resume_from_path

                critic_warmup = critic_warmup_col.number_input(
                    "Critic Warmup Iteration", value=0, min_value=0
                )

                with st.expander("Advanced Config"):
                    default_hdfs_dir = st.text_input("Default HDFS Dir", None)

                    remove_previous_ckpt_in_save_col, del_local_ckpt_after_load_col = st.columns(2)
                    remove_previous_ckpt_in_save = remove_previous_ckpt_in_save_col.checkbox(
                        "Remove Previous Checkpoint in Save", value=False
                    )
                    del_local_ckpt_after_load = del_local_ckpt_after_load_col.checkbox(
                        "Delete Local Checkpoint After Load", value=False
                    )

                    max_actor_ckpt_to_keep_col, max_critic_ckpt_to_keep_col = st.columns(2)
                    max_actor_ckpt_to_keep = max_actor_ckpt_to_keep_col.number_input(
                        "Max Actor Checkpoint to Keep", value=None, min_value=1
                    )
                    max_critic_ckpt_to_keep = max_critic_ckpt_to_keep_col.number_input(
                        "Max Critic Checkpoint to Keep", value=None, min_value=1
                    )

                    total_training_steps = st.number_input(
                        "Total Training Steps", value=None, min_value=1
                    )

            with rl_algorithm_tab:
                st.subheader("RL Algorithm Config")
                gamma_col, lam_col, adv_estimator_col = st.columns(3)
                gamma = gamma_col.number_input("Gamma", value=1.0)
                lam = lam_col.number_input("lam", value=1.0)
                adv_estimator = adv_estimator_col.selectbox(
                    "Advantage Estimator", [member.value for member in AdvantageEstimator], index=0
                )
                kl_penalty_col, kl_ctrl_type_col, kl_ctrl_coef_col = st.columns(3)
                kl_penalty = kl_penalty_col.selectbox(
                    "KL Penalty", ["kl", "abs", "mse", "low_var_kl"], index=0
                )
                kl_ctrl_type = kl_ctrl_type_col.selectbox(
                    "KL Ctrl Type", ["fixed", "adaptive"], index=0
                )
                kl_ctrl_coef = kl_ctrl_coef_col.number_input("KL Ctrl Coef", value=0.001)

            with actor_ref_tab:
                st.subheader("Actor Model Config")
                (
                    actor_ppo_micro_batch_size_per_gpu_col,
                    ref_log_prob_micro_batch_size_per_gpu_col,
                    actor_ulysses_sequence_parallel_size_col,
                ) = st.columns(3)
                actor_ppo_micro_batch_size_per_gpu = (
                    actor_ppo_micro_batch_size_per_gpu_col.number_input(
                        "Micro Batch Size Per GPU for Actor", value=4, min_value=1
                    )
                )
                ref_log_prob_micro_batch_size_per_gpu = (
                    ref_log_prob_micro_batch_size_per_gpu_col.number_input(
                        "Micro Batch Size Per GPU for Ref", value=8, min_value=1
                    )
                )
                actor_ulysses_sequence_parallel_size = (
                    actor_ulysses_sequence_parallel_size_col.number_input(
                        "Ulysses Sequence Parallel Size", value=1, min_value=1, max_value=8
                    )
                )

                (
                    actor_lr_col,
                    actor_grad_clip_col,
                    actor_clip_ratio_col,
                    actor_entropy_coeff_col,
                ) = st.columns(4)
                actor_lr = actor_lr_col.number_input(
                    "Learning Rate for actor",
                    value=1e-6,
                    min_value=1e-7,
                    max_value=1e-3,
                    format="%.1e",
                )
                actor_grad_clip = actor_grad_clip_col.number_input(
                    "Grad Clip", value=1.0, min_value=0.0, max_value=1.0
                )
                actor_clip_ratio = actor_clip_ratio_col.number_input(
                    "Clip Ratio", value=0.2, min_value=0.0, max_value=1.0
                )
                actor_entropy_coeff = actor_entropy_coeff_col.number_input(
                    "Entropy Coeff", value=0.001, min_value=0.0, max_value=1.0
                )

                (
                    actor_lr_warmup_steps_ratio_col,
                    actor_warmup_style_col,
                    actor_alg_type_col,
                ) = st.columns(3)
                actor_lr_warmup_steps_ratio = actor_lr_warmup_steps_ratio_col.number_input(
                    "LR Warmup Steps Ratio", value=0.0, min_value=0.0, max_value=1.0
                )
                actor_warmup_style = actor_warmup_style_col.selectbox(
                    "LR Warmup Style", ["linear", "cosine"], index=0
                )
                if "actor_tau" not in st.session_state:
                    st.session_state.actor_tau = 0.0
                    st.session_state.actor_opmd_baseline = "mean"
                    st.session_state.actor_use_uid = False
                actor_alg_type = actor_alg_type_col.selectbox(
                    "Algorithm Type", ["ppo", "opmd", "pairwise_opmd"], index=0
                )
                if actor_alg_type != "ppo":
                    actor_tau_col, actor_opmd_baseline_col, actor_use_uid_col = st.columns(3)
                    actor_tau = actor_tau_col.number_input(
                        "Tau for OPMD",
                        value=0.0,
                        min_value=0.0,
                        max_value=1.0,
                        format="%.1e",
                    )
                    actor_opmd_baseline = actor_opmd_baseline_col.selectbox(
                        "OPMD Baseline",
                        ["mean", "logavgexp"],
                        index=0,
                    )
                    actor_use_uid = actor_use_uid_col.checkbox("Use UID for OPMD", value=False)
                    st.session_state.actor_tau = actor_tau
                    st.session_state.actor_opmd_baseline = actor_opmd_baseline
                    st.session_state.actor_use_uid = actor_use_uid
                else:
                    actor_tau = st.session_state.actor_tau
                    actor_opmd_baseline = st.session_state.actor_opmd_baseline
                    actor_use_uid = st.session_state.actor_use_uid

                actor_use_kl_loss = st.checkbox("Use KL Loss (True for GRPO)", value=False)
                if "actor_kl_loss_coef" not in st.session_state:
                    st.session_state.actor_kl_loss_coef = 0.001
                    st.session_state.actor_kl_loss_type = "low_var_kl"
                if actor_use_kl_loss:
                    actor_kl_loss_coef_col, actor_kl_loss_type_col = st.columns(2)
                    actor_kl_loss_coef = actor_kl_loss_coef_col.number_input(
                        "KL Loss Coef",
                        value=st.session_state.actor_kl_loss_coef,
                        min_value=0.0,
                        max_value=1.0,
                        format="%.1e",
                    )
                    actor_kl_loss_type_candidates = ["kl", "abs", "mse", "low_var_kl"]
                    actor_kl_loss_type = actor_kl_loss_type_col.selectbox(
                        "KL Loss Type",
                        actor_kl_loss_type_candidates,
                        index=actor_kl_loss_type_candidates.index(
                            st.session_state.actor_kl_loss_type
                        ),
                    )
                    st.session_state.actor_kl_loss_coef = actor_kl_loss_coef
                    st.session_state.actor_kl_loss_type = actor_kl_loss_type
                else:
                    actor_kl_loss_coef = st.session_state.actor_kl_loss_coef
                    actor_kl_loss_type = st.session_state.actor_kl_loss_type

                actor_checkpoint = st.multiselect(
                    "Checkpoint",
                    ["model", "hf_model", "optimizer", "extra"],
                    default=["model", "hf_model", "optimizer", "extra"],
                )

            with critic_tab:
                st.subheader("Critic Model Config")
                (
                    critic_lr_col,
                    critic_grad_clip_col,
                    critic_warmup_style_col,
                    critic_lr_warmup_steps_ratio_col,
                ) = st.columns(4)
                critic_lr = critic_lr_col.number_input(
                    "Learning Rate",
                    key="Learning Rate for Critic",
                    value=1e-6,
                    min_value=1e-7,
                    max_value=1e-3,
                    format="%.1e",
                )
                critic_warmup_style = critic_warmup_style_col.selectbox(
                    "LR Warmup Style",
                    ["constant", "cosine"],
                    index=0,
                )
                critic_lr_warmup_steps_ratio = critic_lr_warmup_steps_ratio_col.number_input(
                    "LR Warmup Steps Ratio",
                    key="LR Warmup Steps Ratio for Critic",
                    value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                )
                critic_grad_clip = critic_grad_clip_col.number_input(
                    "Grad Clip", key="Grad Clip for Critic", value=1.0, min_value=0.0, max_value=1.0
                )

                (
                    critic_cliprange_value_col,
                    critic_ppo_micro_batch_size_per_gpu_col,
                    critic_ulysses_sequence_parallel_size_col,
                ) = st.columns(3)
                critic_cliprange_value = critic_cliprange_value_col.number_input(
                    "Cliprange Value",
                    key="Cliprange Value for Critic",
                    value=0.5,
                    min_value=0.0,
                    max_value=1.0,
                )
                critic_ppo_micro_batch_size_per_gpu = (
                    critic_ppo_micro_batch_size_per_gpu_col.number_input(
                        "Micro Batch Size Per GPU for Critic", value=8, min_value=1
                    )
                )
                critic_ulysses_sequence_parallel_size = (
                    critic_ulysses_sequence_parallel_size_col.number_input(
                        "Ulysses Sequence Parallel Size",
                        key="Ulysses Sequence Parallel Size for Critic",
                        value=1,
                        min_value=1,
                        max_value=8,
                    )
                )

            rollout_node_num = engine_num * tensor_parallel_size // gpu_per_node
            trainer_nnodes = node_num - rollout_node_num
            if node_num == 1:
                trainer_n_gpus_per_node = gpu_per_node - engine_num * tensor_parallel_size
            else:
                trainer_n_gpus_per_node = gpu_per_node
            trainer_config = {
                "data": {
                    "tokenizer": None,
                    "train_files": "placeholder",
                    "val_files": "placeholder",
                    "prompt_key": "placeholder",
                    "max_prompt_length": max_prompt_tokens,
                    "max_response_length": max_response_tokens,
                    "train_batch_size": batch_size_per_gpu * gpu_num * repeat_times,
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
                        "path": model_path,
                        "external_lib": None,
                        "override_config": {},
                        "enable_gradient_checkpointing": enable_gradient_checkpointing,
                        "use_remove_padding": use_remove_padding,
                    },
                    "actor": {
                        "strategy": training_strategy,
                        "ppo_mini_batch_size": batch_size_per_gpu * gpu_num,
                        "ppo_micro_batch_size_per_gpu": actor_ppo_micro_batch_size_per_gpu,
                        "use_dynamic_bsz": use_dynamic_bsz,
                        "ppo_max_token_len_per_gpu": repeat_times
                        * (max_prompt_tokens + max_response_tokens),
                        "grad_clip": actor_grad_clip,
                        "clip_ratio": actor_clip_ratio,
                        "entropy_coeff": actor_entropy_coeff,
                        "use_kl_loss": actor_use_kl_loss,
                        "kl_loss_coef": actor_kl_loss_coef,
                        "kl_loss_type": actor_kl_loss_type,
                        "ppo_epochs": 1,  # TODO
                        "shuffle": False,
                        "ulysses_sequence_parallel_size": actor_ulysses_sequence_parallel_size,
                        "checkpoint": {"contents": actor_checkpoint},
                        "optim": {
                            "lr": actor_lr,
                            "lr_warmup_steps_ratio": actor_lr_warmup_steps_ratio,
                            "warmup_style": actor_warmup_style,
                            "total_training_steps": -1
                            if total_training_steps is None
                            else total_training_steps,
                        },
                        "fsdp_config": fsdp_config,
                        "alg_type": actor_alg_type,
                        "tau": actor_tau,
                        "opmd_baseline": actor_opmd_baseline,
                        "use_uid": actor_use_uid,
                    },
                    "ref": {
                        "fsdp_config": fsdp_config,
                        "log_prob_micro_batch_size_per_gpu": ref_log_prob_micro_batch_size_per_gpu,
                        "log_prob_use_dynamic_bsz": "${actor_rollout_ref.actor.use_dynamic_bsz}",
                        "log_prob_max_token_len_per_gpu": "${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}",
                        "ulysses_sequence_parallel_size": "${actor_rollout_ref.actor.ulysses_sequence_parallel_size}",
                    },
                    "rollout": {
                        "name": "vllm",
                        "temperature": temperature,
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
                        "n": repeat_times,
                    },
                },
                "critic": {
                    "strategy": training_strategy,
                    "optim": {
                        "lr": critic_lr,
                        "lr_warmup_steps_ratio": critic_warmup_style,
                        "warmup_style": critic_lr_warmup_steps_ratio,
                        "total_training_steps": -1
                        if total_training_steps is None
                        else total_training_steps,
                    },
                    "model": {
                        "path": critic_model_path,
                        "tokenizer_path": "${actor_rollout_ref.model.path}",
                        "override_config": {},
                        "external_lib": "${actor_rollout_ref.model.external_lib}",
                        "enable_gradient_checkpointing": enable_gradient_checkpointing,
                        "use_remove_padding": use_remove_padding,
                        "fsdp_config": fsdp_config,
                    },
                    "ppo_mini_batch_size": "${actor_rollout_ref.actor.ppo_mini_batch_size}",
                    "ppo_micro_batch_size_per_gpu": critic_ppo_micro_batch_size_per_gpu,
                    "forward_micro_batch_size_per_gpu": "${critic.ppo_micro_batch_size_per_gpu}",
                    "use_dynamic_bsz": use_dynamic_bsz,
                    "ppo_max_token_len_per_gpu": repeat_times
                    * (max_prompt_tokens + max_response_tokens)
                    * 2,
                    "forward_max_token_len_per_gpu": "${critic.ppo_max_token_len_per_gpu}",
                    "ulysses_sequence_parallel_size": critic_ulysses_sequence_parallel_size,
                    "ppo_epochs": "${actor_rollout_ref.actor.ppo_epochs}",
                    "shuffle": "${actor_rollout_ref.actor.shuffle}",
                    "grad_clip": critic_grad_clip,
                    "cliprange_value": critic_cliprange_value,
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
                    "gamma": gamma,
                    "lam": lam,
                    "adv_estimator": adv_estimator,
                    "kl_penalty": kl_penalty,
                    "kl_ctrl": {"type": kl_ctrl_type, "kl_coef": kl_ctrl_coef},
                },
                "trainer": {
                    "balance_batch": balance_batch,
                    "total_epochs": total_epoch,
                    "project_name": project,
                    "experiment_name": name,
                    "logger": ["wandb"],
                    "val_generations_to_log_to_wandb": 0,
                    "nnodes": trainer_nnodes,
                    "n_gpus_per_node": trainer_n_gpus_per_node,
                    "save_freq": save_freq,
                    "resume_mode": resume_mode,
                    "resume_from_path": resume_from_path,
                    "test_freq": 100,
                    "critic_warmup": critic_warmup,
                    "default_hdfs_dir": default_hdfs_dir,
                    "remove_previous_ckpt_in_save": remove_previous_ckpt_in_save,
                    "del_local_ckpt_after_load": del_local_ckpt_after_load,
                    "default_local_dir": checkpoint_path,
                    "val_before_train": False,
                    "sync_freq": sync_iteration_interval,
                    "max_actor_ckpt_to_keep": max_actor_ckpt_to_keep,
                    "max_critic_ckpt_to_keep": max_critic_ckpt_to_keep,
                },
            }
        else:
            raise ValueError(f"Invalid trainer type: {trainer_type}")

    if st.button("Generate Config", disabled=unfinished_flag):
        config = {
            "data": {
                "total_epochs": total_epoch,
                "batch_size": batch_size_per_gpu * gpu_num,
                "dataset_path": dataset_path,
                "default_workflow_type": default_workflow_type,
                "default_reward_fn_type": default_reward_fn_type,
                "train_split": train_split,
                "eval_split": eval_split,
                "format_config": {
                    "prompt_key": prompt_key,
                    "response_key": response_key,
                },
            },
            "model": {
                "model_path": model_path,
                "max_prompt_tokens": max_prompt_tokens,
                "max_response_tokens": max_response_tokens,
                "checkpoint_path": checkpoint_path,
            },
            "cluster": {
                "node_num": node_num,
                "gpu_per_node": gpu_per_node,
            },
            "buffer": {
                "storage_type": storage_type,
                "db_url": db_url,
                "read_batch_size": batch_size_per_gpu * gpu_num * repeat_times,
                "max_retry_times": max_retry_times,
                "max_retry_interval": max_retry_interval,
            },
            "explorer": {
                "engine_type": engine_type,
                "engine_num": engine_num,
                "runner_num": runner_num,
                "tensor_parallel_size": tensor_parallel_size,
                "enable_prefix_caching": enable_prefix_caching,
                "enforce_eager": enforce_eager,
                "dtype": dtype,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "logprobs": logprobs,
                "repeat_times": repeat_times,
                "backend": backend,
                "max_pending_requests": max_pending_requests,
                "max_waiting_steps": max_waiting_steps,
            },
            "synchronizer": {
                "sync_method": sync_method,
                "sync_iteration_interval": sync_iteration_interval,
            },
            "trainer": {
                "trainer_type": trainer_type,
                "trainer_config_path": trainer_config_path,
                "sft_warmup_iteration": sft_warmup_iteration,
                "eval_interval": eval_interval,
            },
            "monitor": {
                "project": project,
                "name": name,
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
    main()
