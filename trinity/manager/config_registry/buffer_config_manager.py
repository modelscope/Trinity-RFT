import json

import pandas as pd
import streamlit as st

from trinity.buffer.storage import PRIORITY_FUNC
from trinity.common.constants import PromptType, StorageType
from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows import WORKFLOWS
from trinity.manager.config_registry.config_registry import CONFIG_GENERATORS


@CONFIG_GENERATORS.register_config(default_value=20)
def set_total_epochs(**kwargs):
    st.number_input("Total Epochs", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=None)
def set_total_steps(**kwargs):
    st.number_input(
        "Total Steps", min_value=1, help="If set, `Total Epochs` will be ignored", **kwargs
    )


@CONFIG_GENERATORS.register_config(default_value=96)
def set_explore_batch_size(**kwargs):
    st.number_input(
        "Task Batch Size",
        min_value=1,
        help="Number of tasks to explore in one explore step",
        **kwargs,
    )


def get_train_batch_size() -> int:
    return (
        st.session_state["train_batch_size"]
        or st.session_state["explore_batch_size"] * st.session_state["repeat_times"]
    )


def get_train_batch_size_per_gpu() -> int:
    return st.session_state["_train_batch_size_per_gpu"] or max(
        st.session_state["explore_batch_size"]
        * st.session_state["repeat_times"]
        // st.session_state["trainer_gpu_num"],
        1,
    )


def _str_for_train_batch_size():
    trainer_gpu_num_str = (
        "`gpu_per_node * node_num - engine_num * tensor_parallel_size`"
        if st.session_state["mode"] == "both"
        else "`gpu_per_node * node_num`"
    )
    return (
        f"Number of experiences in a mini-batch; defaults to `task_batch_size` * `repeat_times`.\n\n"
        f"Please ensure that `train_batch_size` ({get_train_batch_size()}) can be divided by "
        f"{trainer_gpu_num_str} ({st.session_state['trainer_gpu_num']})."
    )


@CONFIG_GENERATORS.register_config(
    default_value=None,
    visible=lambda: st.session_state["trainer_gpu_num"] > 0,
    other_configs={"_train_batch_size_per_gpu": None},
)
def set_train_batch_size(**kwargs):
    key = kwargs.get("key")
    trainer_gpu_num = st.session_state["trainer_gpu_num"]
    st.session_state[key] = (
        st.session_state["_train_batch_size_per_gpu"] * st.session_state["trainer_gpu_num"]
        if st.session_state["_train_batch_size_per_gpu"] is not None
        else None
    )
    placeholder = st.session_state["explore_batch_size"] * st.session_state["repeat_times"]

    def on_change():
        st.session_state["_train_batch_size_per_gpu"] = max(
            st.session_state[key] // st.session_state["trainer_gpu_num"], 1
        )

    st.number_input(
        "Train Batch Size",
        min_value=trainer_gpu_num,
        step=trainer_gpu_num,
        help=_str_for_train_batch_size(),
        on_change=on_change,
        placeholder=placeholder,
        **kwargs,
    )


@CONFIG_GENERATORS.register_check()
def check_train_batch_size(unfinished_fields: set, key: str):
    if get_train_batch_size() % st.session_state["trainer_gpu_num"] != 0:
        unfinished_fields.add(key)
        st.warning(_str_for_train_batch_size())


@CONFIG_GENERATORS.register_config(default_value=3)
def set_buffer_max_retry_times(**kwargs):
    st.number_input("Max Retry Times", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value=1)
def set_max_retry_interval(**kwargs):
    st.number_input("Max Retry Interval", min_value=1, **kwargs)


@CONFIG_GENERATORS.register_config(default_value="")
def set_taskset_path(**kwargs):
    st.text_input("Taskset Path", **kwargs)


@CONFIG_GENERATORS.register_check()
def check_taskset_path(unfinished_fields: set, key: str):
    if not st.session_state[key].strip():
        unfinished_fields.add(key)
        st.warning("Please input taskset path.")


@CONFIG_GENERATORS.register_config(
    visible=lambda: st.session_state["taskset_path"]
    and "://" not in st.session_state["taskset_path"],
    other_configs={
        "taskset_subset_name": None,
        "taskset_split": "train",
        "taskset_prompt_key": "question",
        "taskset_response_key": "answer",
        "temperature": 1.0,
        "top_p": 1.0,  # TODO: to be used
        "top_k": -1,  # TODO: to be used
        "logprobs": 0,
    },
)
def set_taskset_args(**kwargs):
    subset_name_col, split_col = st.columns(2)
    subset_name_col.text_input(
        "Subset Name :orange-badge[(Needs review)]",
        key="taskset_subset_name",
        help="The subset name used for `datasets.load_datasets`, defaults to `None`",
    )
    split_col.text_input("Train Split :orange-badge[(Needs review)]", key="taskset_split")
    prompt_key_col, response_key_col = st.columns(2)
    prompt_key_col.text_input("Prompt Key :orange-badge[(Needs review)]", key="taskset_prompt_key")
    response_key_col.text_input(
        "Response Key :orange-badge[(Needs review)]", key="taskset_response_key"
    )

    temperature_col, logprobs_col = st.columns(2)
    temperature_col.number_input("Temperature", key="temperature", min_value=0.0, max_value=2.0)
    logprobs_col.number_input("Logprobs", key="logprobs", min_value=0, max_value=20)


def _set_eval_taskset_idx(idx):
    col1, col2 = st.columns([9, 1])
    col1.text_input(
        "Taskset Name",
        key=f"eval_taskset_{idx}_name",
    )
    if col2.button("✖️", key=f"eval_taskset_{idx}_del_flag", type="primary"):
        st.rerun()
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
        help="The subset name used for `datasets.load_datasets`, defaults to `None`",
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

    temperature_col, logprobs_col, n_col = st.columns(3)
    temperature_col.number_input(
        "Temperature",
        key=f"eval_taskset_{idx}_temperature",
        min_value=0.0,
        max_value=1.0,
    )
    logprobs_col.number_input(
        "Logprobs",
        key=f"eval_taskset_{idx}_logprobs",
        min_value=0,
        max_value=20,
    )
    n_col.number_input(
        "Eval repeat times",
        key=f"eval_taskset_{idx}_n",
        min_value=1,
        max_value=20,
    )


@CONFIG_GENERATORS.register_config(other_configs={"_eval_tasksets_num": 0})
def set_eval_tasksets(**kwargs):
    if st.button("Add Eval Taskset"):
        idx = st.session_state["_eval_tasksets_num"]
        st.session_state[f"eval_taskset_{idx}_split"] = "test"
        st.session_state[f"eval_taskset_{idx}_prompt_key"] = "prompt"
        st.session_state[f"eval_taskset_{idx}_response_key"] = "response"
        st.session_state[f"eval_taskset_{idx}_temperature"] = 0.1
        st.session_state["_eval_tasksets_num"] += 1
    if st.session_state["_eval_tasksets_num"] > 0:
        tabs = st.tabs(
            [f"Eval Taskset {i + 1}" for i in range(st.session_state["_eval_tasksets_num"])]
        )
        for idx, tab in enumerate(tabs):
            with tab:
                _set_eval_taskset_idx(idx)


@CONFIG_GENERATORS.register_config(default_value="math_workflow")
def set_default_workflow_type(**kwargs):
    st.selectbox(
        "Default Workflow Type :orange-badge[(Needs review)]",
        WORKFLOWS.modules.keys(),
        help=r"""`simple_workflow`: call 'model.chat()' to get responses.

`math_workflow`: call 'model.chat()' with a pre-defined system prompt to get responses.

Other workflows: conduct multi-turn task for the given dataset.
""",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value="math_workflow")
def set_default_eval_workflow_type(**kwargs):
    st.selectbox(
        "Default Eval Workflow Type :orange-badge[(Needs review)]",
        WORKFLOWS.modules.keys(),
        help=r"""`simple_workflow`: call 'model.chat()' to get responses.

`math_workflow`: call 'model.chat()' with a pre-defined system prompt to get responses.

Other workflows: conduct multi-turn task for the given dataset.
""",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value="math_reward")
def set_default_reward_fn_type(**kwargs):
    st.selectbox(
        "Default Reward Fn Type :orange-badge[(Needs review)]",
        REWARD_FUNCTIONS.modules.keys(),
        help=r"""`accuracy_reward`: check the accuracy for math problems.

`format_reward`: check if the response matches the format (default: `<think>*</think>* <answer>*</answer>`).

`math_reward`: `accuracy_reward` (1 or 0) + `format_reward` (+0.1 or -0.1).
""",
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value=StorageType.QUEUE.value,
    other_configs={
        "_offline_dataset_storage_type": StorageType.FILE.value,
        "_not_offline_dataset_storage_type": StorageType.QUEUE.value,
    },
)
def set_storage_type(**kwargs):
    key = kwargs.get("key")
    if st.session_state["algorithm_type"] in ("dpo", "sft"):
        st.session_state[key] = st.session_state["_offline_dataset_storage_type"]
        storage_candidates = [StorageType.FILE.value, StorageType.SQL.value]
    else:
        st.session_state[key] = st.session_state["_not_offline_dataset_storage_type"]
        storage_candidates = [StorageType.QUEUE.value]

    def on_change():
        if st.session_state["algorithm_type"] not in ("dpo", "sft"):
            st.session_state["_not_offline_dataset_storage_type"] = st.session_state[key]

    st.selectbox(
        "Storage Type",
        storage_candidates,
        on_change=on_change,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(default_value=False)
def set_enable_replay_buffer(**kwargs):
    st.checkbox("Enable Replay Buffer", **kwargs)


@CONFIG_GENERATORS.register_config(
    default_value=None, visible=lambda: st.session_state["enable_replay_buffer"]
)
def set_reuse_cooldown_time(**kwargs):
    st.number_input(
        "Reuse Cooldown Time",
        min_value=0.0,
        max_value=1e5,
        help="Leave blank to indicate no experience reuse",
        placeholder=None,
        **kwargs,
    )


@CONFIG_GENERATORS.register_config(
    default_value="linear_decay", visible=lambda: st.session_state["enable_replay_buffer"]
)
def set_priority_fn(**kwargs):
    candidates = list(PRIORITY_FUNC.modules.keys())
    st.selectbox(
        "Priority Function",
        candidates,
        **kwargs,
    )


def parse_priority_fn_args(raw_data: str):
    try:
        data = json.loads(raw_data)
        if data["priority_fn"] != st.session_state["priority_fn"]:
            raise ValueError
        return data["fn_args"]
    except (json.JSONDecodeError, KeyError, ValueError):
        print(f"Use `default_config` for {st.session_state['priority_fn']}")
        return PRIORITY_FUNC.get(st.session_state["priority_fn"]).default_config()


@CONFIG_GENERATORS.register_config(
    default_value="", visible=lambda: st.session_state["enable_replay_buffer"]
)
def set_priority_fn_args(**kwargs):
    key = kwargs.get("key")
    df = pd.DataFrame([parse_priority_fn_args(st.session_state[key])])
    df.index = [st.session_state["priority_fn"]]
    st.caption("Priority Function Args")
    df = st.data_editor(df)
    st.session_state[key] = json.dumps(
        {
            "fn_args": df.to_dict(orient="records")[0],
            "priority_fn": st.session_state["priority_fn"],
        }
    )


@CONFIG_GENERATORS.register_config(
    default_value="",
    other_configs={
        "_offline_dataset_experience_buffer_path": "",
        "_not_offline_dataset_experience_buffer_path": "",
    },
)
def set_experience_buffer_path(**kwargs):
    key = kwargs.get("key")
    if st.session_state["algorithm_type"] in ("dpo", "sft"):
        if (
            st.session_state["taskset_path"]
            and not st.session_state["_offline_dataset_experience_buffer_path"]
        ):
            st.session_state["_offline_dataset_experience_buffer_path"] = st.session_state[
                "taskset_path"
            ]
        st.session_state[key] = st.session_state["_offline_dataset_experience_buffer_path"]
        title = "Dataset Path"
        help_msg = r"""Path to the dataset."""
    else:
        st.session_state[key] = st.session_state["_not_offline_dataset_experience_buffer_path"]
        title = "Experience Buffer Path"
        help_msg = r"""This path is used for experiences persistent storage, default to `None`."""

    def on_change():
        if st.session_state["algorithm_type"] in ("dpo", "sft"):
            st.session_state["_offline_dataset_experience_buffer_path"] = st.session_state[key]
        else:
            st.session_state["_not_offline_dataset_experience_buffer_path"] = st.session_state[key]

    st.text_input(title, help=help_msg, on_change=on_change, **kwargs)


@CONFIG_GENERATORS.register_check()
def check_experience_buffer_path(unfinished_fields: set, key: str):
    if st.session_state["algorithm_type"] == "dpo":
        if not st.session_state[key].strip():
            unfinished_fields.add(key)
            st.warning("Please input DPO dataset path.")
    elif st.session_state["algorithm_type"] == "sft":
        if not st.session_state[key].strip():
            unfinished_fields.add(key)
            st.warning("Please input SFT dataset path.")


@CONFIG_GENERATORS.register_config(
    other_configs={
        "dpo_dataset_subset_name": None,
        "dpo_dataset_train_split": "train",
        "dpo_dataset_prompt_type": PromptType.PLAINTEXT.value,
        "dpo_dataset_prompt_key": "prompt",
        "dpo_dataset_chosen_key": "chosen",
        "dpo_dataset_rejected_key": "rejected",
    }
)
def set_dpo_dataset_kwargs(**kwargs):
    (
        dpo_dataset_subset_name_col,
        dpo_dataset_train_split_col,
        dpo_dataset_prompt_type_col,
    ) = st.columns(3)
    dpo_dataset_subset_name_col.text_input(
        "Subset Name :orange-badge[(Needs review)]", key="dpo_dataset_subset_name"
    )
    dpo_dataset_train_split_col.text_input(
        "Train Split :orange-badge[(Needs review)]", key="dpo_dataset_train_split"
    )
    dpo_dataset_prompt_type_col.selectbox(
        "Prompt Type :orange-badge[(Needs review)]",
        [prompt_type.value for prompt_type in PromptType],
        key="dpo_dataset_prompt_type",
    )

    (
        dpo_dataset_prompt_key_col,
        dpo_dataset_chosen_key_col,
        dpo_dataset_rejected_key_col,
    ) = st.columns(3)
    dpo_dataset_prompt_key_col.text_input(
        "Prompt Key :orange-badge[(Needs review)]", key="dpo_dataset_prompt_key"
    )
    dpo_dataset_chosen_key_col.text_input(
        "Chosen Key :orange-badge[(Needs review)]", key="dpo_dataset_chosen_key"
    )
    dpo_dataset_rejected_key_col.text_input(
        "Rejected Key :orange-badge[(Needs review)]",
        key="dpo_dataset_rejected_key",
    )


@CONFIG_GENERATORS.register_config(
    other_configs={
        "sft_dataset_subset_name": None,
        "sft_dataset_train_split": "train",
        "sft_dataset_prompt_type": PromptType.MESSAGES.value,
        "sft_dataset_prompt_key": "prompt",
        "sft_dataset_response_key": "response",
        "sft_dataset_messages_key": "messages",
    }
)
def set_sft_dataset_kwargs(**kwargs):
    (
        sft_dataset_subset_name_col,
        sft_dataset_train_split_col,
        sft_dataset_prompt_type_col,
    ) = st.columns(3)
    sft_dataset_subset_name_col.text_input(
        "Subset Name :orange-badge[(Needs review)]", key="sft_dataset_subset_name"
    )
    sft_dataset_train_split_col.text_input(
        "Train Split :orange-badge[(Needs review)]", key="sft_dataset_train_split"
    )
    sft_dataset_prompt_type_col.selectbox(
        "Prompt Type :orange-badge[(Needs review)]",
        [prompt_type.value for prompt_type in PromptType],
        key="sft_dataset_prompt_type",
        help="When `Prompt Type` is `plaintext`, `Prompt Key` and `Response Key` are effective; when `Prompt Type` is `messages`, `Messages Key` is effective.",
    )

    (
        sft_dataset_prompt_key_col,
        sft_dataset_response_key_col,
        sft_dataset_messages_key_col,
    ) = st.columns(3)
    sft_dataset_prompt_key_col.text_input(
        "Prompt Key :orange-badge[(Needs review)]", key="sft_dataset_prompt_key"
    )
    sft_dataset_response_key_col.text_input(
        "Response Key :orange-badge[(Needs review)]", key="sft_dataset_response_key"
    )
    sft_dataset_messages_key_col.text_input(
        "Messages Key :orange-badge[(Needs review)]",
        key="sft_dataset_messages_key",
    )
