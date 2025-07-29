import streamlit as st

from trinity.algorithm.algorithm import ALGORITHM_TYPE
from trinity.common.constants import SyncMethod
from trinity.manager.config_registry.config_registry import CONFIG_GENERATORS


def use_critic():
    algorithm = ALGORITHM_TYPE.get(st.session_state["algorithm_type"])
    return algorithm.use_critic


@CONFIG_GENERATORS.register_config(default_value="verl")
def set_trainer_type(**kwargs):
    st.selectbox("Trainer Type", ["verl"], **kwargs)


@CONFIG_GENERATORS.register_config(default_value=100, other_configs={"_nccl_save_interval": 100})
def set_save_interval(**kwargs):
    key = kwargs.get("key")
    if (
        st.session_state["algorithm_type"] == "dpo"
        or st.session_state["sync_method"] == SyncMethod.NCCL.value
    ):
        st.session_state[key] = st.session_state["_nccl_save_interval"]
        freeze_save_interval = False
    else:
        st.session_state[key] = st.session_state["sync_interval"]
        freeze_save_interval = True

    def on_change():
        if (
            st.session_state["algorithm_type"] == "dpo"
            or st.session_state["sync_method"] == SyncMethod.NCCL.value
        ):
            st.session_state["_nccl_save_interval"] = st.session_state[key]

    st.number_input(
        "Save Interval",
        min_value=1,
        help="Set to `sync_interval` when `algorithm_type != DPO && sync_method == checkpoint`",
        disabled=freeze_save_interval,
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=True)
def set_enable_preview(**kwargs):
    st.checkbox("Enable Preview", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=1.0)
def set_actor_grad_clip(**kwargs):
    st.number_input(
        "Grad Clip :blue-badge[(Actor)]",
        min_value=0.0,
        max_value=1.0,
        help="Clipping by Norm",
        **kwargs,
    )


# veRL Trainer Configs


@CONFIG_GENERATORS.register_config(
    default_value=[
        "balance_batch",
        "gradient_checkpointing",
        "remove_padding",
        "dynamic_bsz",
    ]
)
def set_training_args(**kwargs):
    st.multiselect(
        "Training Args",
        [
            "balance_batch",
            "gradient_checkpointing",
            "remove_padding",
            "dynamic_bsz",
            "use_fused_kernels",
        ],
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=1)
def set_ppo_epochs(**kwargs):
    st.number_input("PPO Epochs", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value="fsdp")
def set_training_strategy(**kwargs):
    st.selectbox(
        "Training Strategy",
        ["fsdp", "megatron"],
        help="megatron is not tested",
        **kwargs,
    )


def use_fsdp():
    return st.session_state["training_strategy"] == "fsdp"


@CONFIG_GENERATORS.register_config(default_value=False, visible=use_fsdp)
def set_param_offload(**kwargs):
    st.checkbox("FSDP Param Offload", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False, visible=use_fsdp)
def set_optimizer_offload(**kwargs):
    st.checkbox("FSDP Optimizer Offload", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False, visible=use_fsdp)
def set_forward_prefetch(**kwargs):
    st.checkbox("FSDP Forward Prefetch", **kwargs)


@CONFIG_GENERATORS.register_config(default_value="auto")
def set_resume_mode(**kwargs):
    st.selectbox("Resume Mode", ["disable", "auto", "resume_path"], **kwargs)


@CONFIG_GENERATORS.register_config(
    default_value="", visible=lambda: st.session_state["resume_mode"] == "resume_path"
)
def set_resume_from_path(**kwargs):
    st.text_input("Resume Path", **kwargs)


@CONFIG_GENERATORS.register_check()
def check_resume_from_path(unfinished_fields: set, key: str):
    if st.session_state["resume_mode"] == "resume_path" and (
        not st.session_state[key].strip() or "global_step_" not in st.session_state[key]
    ):
        unfinished_fields.add(key)
        st.warning("Please input a valid resume path when `resume_mode == resume_path`")


@CONFIG_GENERATORS.register_config(
    default_value="triton", visible=lambda: "use_fused_kernels" in st.session_state["training_args"]
)
def set_impl_backend(**kwargs):
    st.selectbox(
        "Impl Backend",
        ["torch", "triton"],
        help="Backend For FusedKernel",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=0)
def set_critic_warmup(**kwargs):
    st.number_input("Critic Warmup Steps", min_value=0, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=None)
def set_total_training_steps(**kwargs):
    st.number_input("Total Training Steps", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=None)
def set_default_hdfs_dir(**kwargs):
    st.text_input("Default HDFS Dir", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False)
def set_remove_previous_ckpt_in_save(**kwargs):
    st.checkbox("Remove Previous Checkpoint in Save", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False)
def set_del_local_ckpt_after_load(**kwargs):
    st.checkbox("Delete Local Checkpoint After Load", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=None)
def set_max_actor_ckpt_to_keep(**kwargs):
    st.number_input("Max Actor Checkpoint to Keep", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=None)
def set_max_critic_ckpt_to_keep(**kwargs):
    st.number_input("Max Critic Checkpoint to Keep", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=True)
def set_norm_adv_by_std_in_grpo(**kwargs):
    st.checkbox("Norm Adv by Std in GRPO", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False)
def set_use_kl_in_reward(**kwargs):
    st.checkbox("Use KL in Reward", **kwargs)


@CONFIG_GENERATORS.register_config(default_value="low_var_kl")
def set_kl_penalty(**kwargs):
    st.selectbox("KL Penalty", ["kl", "abs", "mse", "low_var_kl"], **kwargs)


@CONFIG_GENERATORS.register_config(default_value="fixed")
def set_kl_ctrl_type(**kwargs):
    st.selectbox("KL Ctrl Type", ["fixed", "adaptive"], **kwargs)


@CONFIG_GENERATORS.register_config(default_value=0.001)
def set_kl_ctrl_coef(**kwargs):
    st.number_input("KL Ctrl Coef", format="%.1e", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=10000)
def set_horizon(**kwargs):
    st.number_input("Horizon", min_value=1.0, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=0.1)
def set_target_kl(**kwargs):
    st.number_input("Target KL", format="%.1e", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=4)
def set_actor_ppo_micro_batch_size_per_gpu(**kwargs):
    key = kwargs.get("key")
    max_value = st.session_state["_train_batch_size_per_gpu"]
    st.session_state[key] = min(st.session_state[key], max_value)
    st.number_input(
        "Micro Batch Size Per GPU :blue-badge[(Actor)]", min_value=1, max_value=max_value, **kwargs
    )


@CONFIG_GENERATORS.register_config(default_value=8)
def set_ref_log_prob_micro_batch_size_per_gpu(**kwargs):
    key = kwargs.get("key")
    max_value = st.session_state["_train_batch_size_per_gpu"]
    st.session_state[key] = min(st.session_state[key], max_value)
    st.number_input(
        "Micro Batch Size Per GPU :blue-badge[(Ref)]", min_value=1, max_value=max_value, **kwargs
    )


@CONFIG_GENERATORS.register_config(default_value=1)
def set_actor_ulysses_sequence_parallel_size(**kwargs):
    st.number_input(
        "Ulysses Sequence Parallel Size",
        min_value=1,
        max_value=8,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=False)
def set_actor_entropy_from_logits_with_chunking(**kwargs):
    st.checkbox("Entropy from Logits with Chunking", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False)
def set_actor_entropy_checkpointing(**kwargs):
    st.checkbox("Entropy Checkpointing", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=1e-6)
def set_actor_lr(**kwargs):
    st.number_input(
        "Learning Rate :blue-badge[(Actor)]",
        min_value=1e-7,
        max_value=1e-3,
        format="%.1e",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value="constant")
def set_actor_warmup_style(**kwargs):
    st.selectbox(
        "LR Warmup Style :blue-badge[(Actor)]",
        ["constant", "cosine"],
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=0.0)
def set_actor_lr_warmup_steps_ratio(**kwargs):
    st.number_input(
        "LR Warmup Steps Ratio :blue-badge[(Actor)]",
        min_value=0.0,
        max_value=1.0,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=["model", "hf_model", "optimizer", "extra"])
def set_actor_checkpoint(**kwargs):
    st.multiselect(
        "Checkpoint",
        ["model", "hf_model", "optimizer", "extra"],
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=1e-6, visible=use_critic)
def set_critic_lr(**kwargs):
    st.number_input(
        "Learning Rate :blue-badge[(Critic)]",
        min_value=1e-7,
        max_value=1e-3,
        format="%.1e",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value="constant", visible=use_critic)
def set_critic_warmup_style(**kwargs):
    st.selectbox(
        "LR Warmup Style :blue-badge[(Critic)]",
        ["constant", "cosine"],
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=0.0, visible=use_critic)
def set_critic_lr_warmup_steps_ratio(**kwargs):
    st.number_input(
        "LR Warmup Steps Ratio :blue-badge[(Critic)]",
        min_value=0.0,
        max_value=1.0,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=1.0, visible=use_critic)
def set_critic_grad_clip(**kwargs):
    st.number_input(
        "Grad Clip :blue-badge[(Critic)]",
        min_value=0.0,
        max_value=1.0,
        help="Clipping by Norm",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=0.5, visible=use_critic)
def set_critic_cliprange_value(**kwargs):
    st.number_input(
        "Cliprange Value",
        min_value=0.0,
        max_value=1.0,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=8, visible=use_critic)
def set_critic_ppo_micro_batch_size_per_gpu(**kwargs):
    key = kwargs.get("key")
    max_value = st.session_state["_train_batch_size_per_gpu"]
    st.session_state[key] = min(st.session_state[key], max_value)
    st.number_input(
        "Micro Batch Size Per GPU :blue-badge[(Critic)]",
        min_value=1,
        max_value=max_value,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=1, visible=use_critic)
def set_critic_ulysses_sequence_parallel_size(**kwargs):
    st.number_input(
        "Ulysses Sequence Parallel Size",
        min_value=1,
        max_value=8,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=["model", "optimizer", "extra"], visible=use_critic
)
def set_critic_checkpoint(**kwargs):
    st.multiselect(
        "Checkpoint",
        ["model", "hf_model", "optimizer", "extra"],
        **kwargs,
    )
