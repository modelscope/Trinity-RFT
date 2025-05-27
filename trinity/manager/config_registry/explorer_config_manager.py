import streamlit as st

from trinity.common.constants import AlgorithmType, SyncMethod
from trinity.manager.config_registry.config_registry import CONFIG_GENERATORS
from trinity.manager.config_registry.model_config_manager import set_trainer_gpu_num


def explorer_condition() -> bool:
    return st.session_state["mode"] == "both"


@CONFIG_GENERATORS.register_config(default_value=32, condition=explorer_condition)
def set_runner_num(**kwargs):
    st.number_input("Runner Num", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=900, condition=explorer_condition)
def set_max_timeout(**kwargs):
    st.number_input("Max Timeout", min_value=0, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=2, condition=explorer_condition)
def set_explorer_max_retry_times(**kwargs):
    st.number_input("Explorer Max Retry Times", min_value=0, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=1000, condition=explorer_condition)
def set_eval_interval(**kwargs):
    st.number_input("Eval Interval", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=True, condition=explorer_condition)
def set_eval_on_latest_checkpoint(**kwargs):
    st.checkbox("Eval on Latest Checkpoint", **kwargs)


@CONFIG_GENERATORS.register_config(default_value="vllm_async", condition=explorer_condition)
def set_engine_type(**kwargs):
    st.selectbox("Engine Type", ["vllm_async", "vllm"], **kwargs)


def _str_for_engine_num_and_tp_size():
    return r"""and it must meet the following constraints:
```python
assert engine_num * tensor_parallel_size < gpu_per_node * node_num
if node_num > 1:
assert gpu_per_node % tensor_parallel_size == 0
assert engine_num * tensor_parallel_size % gpu_per_node == 0
```"""


@CONFIG_GENERATORS.register_config(default_value=2, condition=explorer_condition)
def set_engine_num(**kwargs):
    key = kwargs.get("key")
    total_gpu_num = st.session_state["total_gpu_num"]
    max_engine_num = (total_gpu_num - 1) // st.session_state["tensor_parallel_size"]
    if st.session_state[key] > max_engine_num:
        st.session_state[key] = max_engine_num
        set_trainer_gpu_num()
    st.number_input(
        "Engine Num",
        min_value=1,
        max_value=max_engine_num,
        help=f"`engine_num` is used to set the quantity of inference engines, "
        f"{_str_for_engine_num_and_tp_size()}",
        on_change=set_trainer_gpu_num,
        **kwargs,
    )


@CONFIG_GENERATORS.register_check()
def check_engine_num(unfinished_fields: set, key: str):
    node_num = st.session_state["node_num"]
    gpu_per_node = st.session_state["gpu_per_node"]
    engine_num = st.session_state["engine_num"]
    tensor_parallel_size = st.session_state["tensor_parallel_size"]
    if node_num > 1:
        if engine_num * tensor_parallel_size % gpu_per_node != 0:
            unfinished_fields.add("engine_num")
            st.warning(
                "Please ensure that `engine_num * tensor_parallel_size` can be divided by `gpu_per_node` when `node_num > 1`."
            )


@CONFIG_GENERATORS.register_config(default_value=1, condition=explorer_condition)
def set_tensor_parallel_size(**kwargs):
    key = kwargs.get("key")
    total_gpu_num = st.session_state["total_gpu_num"]
    max_tensor_parallel_size = (total_gpu_num - 1) // st.session_state["engine_num"]
    if st.session_state[key] > max_tensor_parallel_size:
        st.session_state[key] = max_tensor_parallel_size
        set_trainer_gpu_num()
    st.number_input(
        "Tensor Parallel Size",
        min_value=1,
        max_value=max_tensor_parallel_size,
        help=f"`tensor_parallel_size` is used to set the tensor parallel size of inference engines, "
        f"{_str_for_engine_num_and_tp_size()}",
        on_change=set_trainer_gpu_num,
        **kwargs,
    )


@CONFIG_GENERATORS.register_check()
def check_tensor_parallel_size(unfinished_fields: set, key: str):
    node_num = st.session_state["node_num"]
    gpu_per_node = st.session_state["gpu_per_node"]
    tensor_parallel_size = st.session_state["tensor_parallel_size"]
    if node_num > 1:
        if gpu_per_node % tensor_parallel_size != 0:
            unfinished_fields.add("tensor_parallel_size")
            st.warning(
                "Please ensure that `tensor_parallel_size` is a factor of `gpu_per_node` when `node_num > 1`."
            )


@CONFIG_GENERATORS.register_config(default_value=True, condition=explorer_condition)
def set_use_v1(**kwargs):
    st.checkbox("Use V1 Engine", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=True, condition=explorer_condition)
def set_enforce_eager(**kwargs):
    st.checkbox("Enforce Eager", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False, condition=explorer_condition)
def set_enable_prefix_caching(**kwargs):
    st.checkbox("Prefix Caching", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False, condition=explorer_condition)
def set_enable_chunked_prefill(**kwargs):
    st.checkbox("Chunked Prefill", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=0.9, condition=explorer_condition)
def set_gpu_memory_utilization(**kwargs):
    st.number_input("GPU Memory Utilization", min_value=0.0, max_value=1.0, **kwargs)


@CONFIG_GENERATORS.register_config(default_value="bfloat16", condition=explorer_condition)
def set_dtype(**kwargs):
    st.selectbox("Dtype", ["bfloat16", "float16", "float32"], **kwargs)


@CONFIG_GENERATORS.register_config(default_value=42, condition=explorer_condition)
def set_seed(**kwargs):
    st.number_input("Seed", step=1, **kwargs)


# TODO: max_prompt_tokens
# TODO: max_response_tokens
# TODO: chat_template


@CONFIG_GENERATORS.register_config(default_value=False, condition=explorer_condition)
def set_enable_thinking(**kwargs):
    st.checkbox("Enable Thinking For Qwen3", **kwargs)


@CONFIG_GENERATORS.register_config(default_value=False, condition=explorer_condition)
def set_enable_openai_api(**kwargs):
    st.checkbox("Enable OpenAI API", **kwargs)


# TODO: Auxiliary Models Configs
def _set_auxiliary_model_idx(idx):
    col1, col2 = st.columns([9, 1])
    col1.text_input(
        "Model Path",
        key=f"auxiliary_model_{idx}_model_path",
    )
    if col2.button("✖️", key=f"auxiliary_model_{idx}_del_flag", type="primary"):
        st.rerun()

    engine_type_col, engine_num_col, tensor_parallel_size_col = st.columns(3)
    total_gpu_num = st.session_state["total_gpu_num"]
    max_engine_num = (total_gpu_num - 1) // st.session_state["tensor_parallel_size"]
    max_tensor_parallel_size = (total_gpu_num - 1) // st.session_state["engine_num"]
    engine_type_col.selectbox(
        "Engine Type", ["vllm_async"], key=f"auxiliary_model_{idx}_engine_type"
    )
    engine_num_col.number_input(
        "Engine Num",
        min_value=1,
        max_value=max_engine_num,
        help=f"`engine_num` is used to set the quantity of inference engines, "
        f"{_str_for_engine_num_and_tp_size()}",
        on_change=set_trainer_gpu_num,
        key=f"auxiliary_model_{idx}_engine_num",
    )
    tensor_parallel_size_col.number_input(
        "Tensor Parallel Size",
        min_value=1,
        max_value=max_tensor_parallel_size,
        help=f"`tensor_parallel_size` is used to set the tensor parallel size of inference engines, "
        f"{_str_for_engine_num_and_tp_size()}",
        on_change=set_trainer_gpu_num,
        key=f"auxiliary_model_{idx}_tensor_parallel_size",
    )

    gpu_memory_utilization_col, dtype_col, seed_col = st.columns(3)
    gpu_memory_utilization_col.number_input(
        "GPU Memory Utilization",
        min_value=0.0,
        max_value=1.0,
        key=f"auxiliary_model_{idx}_gpu_memory_utilization",
    )
    dtype_col.selectbox(
        "Dtype", ["bfloat16", "float16", "float32"], key=f"auxiliary_model_{idx}_dtype"
    )
    seed_col.number_input("Seed", step=1, key=f"auxiliary_model_{idx}_seed")

    (
        use_v1_col,
        enforce_eager_col,
        enable_prefix_caching_col,
        enable_chunked_prefill_col,
    ) = st.columns(4)
    use_v1_col.checkbox("Use V1 Engine", key=f"auxiliary_model_{idx}_use_v1")
    enforce_eager_col.checkbox("Enforce Eager", key=f"auxiliary_model_{idx}_enforce_eager")
    enable_prefix_caching_col.checkbox(
        "Prefix Caching", key=f"auxiliary_model_{idx}_enable_prefix_caching"
    )
    enable_chunked_prefill_col.checkbox(
        "Chunked Prefill", key=f"auxiliary_model_{idx}_enable_chunked_prefill"
    )

    enable_thinking_col, enable_openai_api = st.columns(2)
    enable_thinking_col.checkbox(
        "Enable Thinking For Qwen3", key=f"auxiliary_model_{idx}_enable_thinking"
    )
    enable_openai_api.checkbox("Enable OpenAI API", key=f"auxiliary_model_{idx}_enable_openai_api")


@CONFIG_GENERATORS.register_config(other_configs={"_auxiliary_models_num": 0})
def set_auxiliary_models(**kwargs):
    if st.button("Add Auxiliary Models"):
        st.session_state["_auxiliary_models_num"] += 1
    if st.session_state["_auxiliary_models_num"] > 0:
        tabs = st.tabs(
            [f"Auxiliary Model {i + 1}" for i in range(st.session_state["_auxiliary_models_num"])]
        )
        for idx, tab in enumerate(tabs):
            with tab:
                _set_auxiliary_model_idx(idx)


# Synchronizer Configs


@CONFIG_GENERATORS.register_config(
    default_value=SyncMethod.NCCL.value,
    condition=explorer_condition,
    other_configs={"_not_dpo_sync_method": SyncMethod.NCCL.value},
)
def set_sync_method(**kwargs):
    key = kwargs.get("key")
    if st.session_state["algorithm_type"] == AlgorithmType.DPO.value:
        st.session_state[key] = SyncMethod.CHECKPOINT.value
        disabled = True
    else:
        st.session_state[key] = st.session_state["_not_dpo_sync_method"]
        disabled = False

    def on_change():
        if st.session_state["algorithm_type"] != AlgorithmType.DPO.value:
            st.session_state["_not_dpo_sync_method"] = st.session_state[key]

    st.selectbox(
        "Sync Method",
        [sync_method.value for sync_method in SyncMethod],
        help="""`nccl`: the explorer and trainer sync model weights once every `sync_interval` steps.

`checkpoint`: the trainer saves the model checkpoint, and the explorer loads it at `sync_interval`.""",
        disabled=disabled,
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=10, condition=explorer_condition)
def set_sync_interval(**kwargs):
    st.number_input(
        "Sync Interval",
        min_value=1,
        help="""The step interval at which the `explorer` and `trainer` synchronize model weight.""",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=1200, condition=explorer_condition)
def set_sync_timeout(**kwargs):
    st.number_input(
        "Sync Timeout",
        min_value=1,
        help="The timeout value for the synchronization operation.",
        **kwargs,
    )
