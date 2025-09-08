import os

import streamlit as st

from trinity.manager.config_registry.config_registry import CONFIG_GENERATORS
from trinity.manager.config_registry.trainer_config_manager import use_critic
from trinity.utils.monitor import MONITOR


def set_total_gpu_num():
    st.session_state["total_gpu_num"] = (
        st.session_state["gpu_per_node"] * st.session_state["node_num"]
    )
    set_trainer_gpu_num()


def set_trainer_gpu_num():
    if st.session_state["mode"] == "both":
        trainer_gpu_num = (
            st.session_state["total_gpu_num"]
            - st.session_state["engine_num"] * st.session_state["tensor_parallel_size"]
        )
        for idx in range(st.session_state["_auxiliary_models_num"]):
            engine_num = st.session_state[f"auxiliary_model_{idx}_engine_num"]
            tensor_parallel_size = st.session_state[f"auxiliary_model_{idx}_tensor_parallel_size"]
            trainer_gpu_num -= engine_num * tensor_parallel_size
        st.session_state["trainer_gpu_num"] = trainer_gpu_num
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


@CONFIG_GENERATORS.register_config(default_value="tensorboard")
def set_monitor_type(**kwargs):
    st.selectbox(
        "Monitor Type",
        options=MONITOR.modules.keys(),
        **kwargs,
    )


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
    visible=use_critic,
)
def set_critic_model_path(**kwargs):
    st.text_input(
        "Critic Model Path (defaults to `model_path`)",
        key="critic_model_path",
    )


@CONFIG_GENERATORS.register_config(default_value=None)
def set_max_prompt_tokens(**kwargs):
    st.number_input("Max Prompt Length", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=1)
def set_min_response_tokens(**kwargs):
    st.number_input("Min Response Length", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=1024)
def set_max_response_tokens(**kwargs):
    st.number_input("Max Response Length", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=2048)
def set_max_model_len(**kwargs):
    st.number_input("Max Model Length", min_value=1, **kwargs)


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
