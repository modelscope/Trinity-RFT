import os

import streamlit as st

from trinity.common.constants import AlgorithmType, MonitorType
from trinity.manager.config_registry.config_registry import CONFIG_GENERATORS
from trinity.trainer.verl.ray_trainer import AdvantageEstimator


def set_total_gpu_num():
    st.session_state["total_gpu_num"] = (
        st.session_state["gpu_per_node"] * st.session_state["node_num"]
    )
    set_trainer_gpu_num()


def set_trainer_gpu_num():
    if st.session_state["mode"] == "both":
        st.session_state["trainer_gpu_num"] = (
            st.session_state["total_gpu_num"]
            - st.session_state["engine_num"] * st.session_state["tensor_parallel_size"]
        )
    else:  # model == train
        st.session_state["trainer_gpu_num"] = st.session_state["total_gpu_num"]


@CONFIG_GENERATORS.register_config(default_value="Trinity-RFT")
def set_project(**kwargs):
    st.text_input("Project", **kwargs)


@CONFIG_GENERATORS.register_config(default_value="qwen2.5-1.5B")
def set_exp_name(**kwargs):
    st.text_input("Experiment Name", **kwargs)


@CONFIG_GENERATORS.register_config(default_value="")
def set_checkpoint_root_dir(**kwargs):
    st.text_input("Checkpoint Root Dir", **kwargs)


@CONFIG_GENERATORS.register_check()
def check_checkpoint_root_dir(unfinished_fields: set, key: str):
    if not st.session_state[key].strip():  # TODO: may auto generate
        unfinished_fields.add(key)
        st.warning("Please input checkpoint root dir.")
    elif not os.path.isabs(st.session_state[key].strip()):
        unfinished_fields.add("checkpoint_root_dir")
        st.warning("Please input an absolute path.")


@CONFIG_GENERATORS.register_config(default_value=MonitorType.TENSORBOARD.value)
def set_monitor_type(**kwargs):
    st.selectbox(
        "Monitor Type",
        options=[monitor_type.value for monitor_type in MonitorType],
        **kwargs,
    )


# Algorithm Configs


@CONFIG_GENERATORS.register_config(
    default_value=AlgorithmType.PPO.value,
    other_configs={"mode": "both", "adv_estimator": AdvantageEstimator.GAE.value},
)
def set_algorithm_type(**kwargs):
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
        set_trainer_gpu_num()

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


@CONFIG_GENERATORS.register_config(
    default_value=1,
    condition=lambda: st.session_state["mode"] == "both",
    other_configs={
        "_grouped_adv_repeat_times": 2,
        "_not_grouped_adv_repeat_times": 1,
    },
)
def set_repeat_times(**kwargs):  # TODO
    key = kwargs.get("key")
    grouped_adv_algorithms = [
        AlgorithmType.GRPO.value,
        AlgorithmType.OPMD.value,  # TODO: may add rloo
    ]
    if st.session_state["algorithm_type"] in grouped_adv_algorithms:
        min_repeat_times = 2
        st.session_state[key] = st.session_state["_grouped_adv_repeat_times"]
    else:
        min_repeat_times = 1
        st.session_state[key] = st.session_state["_not_grouped_adv_repeat_times"]

    def on_change():
        if st.session_state["algorithm_type"] in grouped_adv_algorithms:
            st.session_state["_grouped_adv_repeat_times"] = st.session_state[key]
        else:
            st.session_state["_not_grouped_adv_repeat_times"] = st.session_state[key]

    st.number_input(
        "Repeat Times",
        min_value=min_repeat_times,
        help="`repeat_times` is used to set how many experiences each task can generate, "
        "and it must be greater than `1` when `algorithm_type` is `opmd` or `grpo`.",
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=1.0)
def set_gamma(**kwargs):
    st.number_input(r"Gamma :blue-badge[$\gamma$]", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=1.0)
def set_lam(**kwargs):
    st.number_input(r"Lambda :blue-badge[$\lambda$]", **kwargs)


# Model Configs


@CONFIG_GENERATORS.register_config(default_value="")
def set_model_path(**kwargs):
    st.text_input("Model Path", **kwargs)


@CONFIG_GENERATORS.register_check()
def check_model_path(unfinished_fields: set, key: str):
    if not st.session_state[key].strip():
        unfinished_fields.add(key)
        st.warning("Please input model path.")


@CONFIG_GENERATORS.register_config(
    default_value="",
    condition=lambda: st.session_state["adv_estimator"] == AdvantageEstimator.GAE.value,
)
def set_critic_model_path(**kwargs):
    st.text_input(
        "Critic Model Path (defaults to `model_path`)",
        key="critic_model_path",
    )


@CONFIG_GENERATORS.register_config(default_value=1024)
def set_max_prompt_tokens(**kwargs):
    st.number_input("Max Prompt Tokens", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=1024)
def set_max_response_tokens(**kwargs):
    st.number_input("Max Response Tokens", min_value=1, **kwargs)


# Cluster Config


@CONFIG_GENERATORS.register_config(default_value=1)
def set_node_num(**kwargs):
    st.number_input("Node Num", min_value=1, on_change=set_total_gpu_num, **kwargs)


@CONFIG_GENERATORS.register_config(
    default_value=8, other_configs={"total_gpu_num": 8, "trainer_gpu_num": 6}
)
def set_gpu_per_node(**kwargs):
    st.number_input(
        "GPU Per Node",
        min_value=1,
        max_value=8,
        on_change=set_total_gpu_num,
        **kwargs,
    )
