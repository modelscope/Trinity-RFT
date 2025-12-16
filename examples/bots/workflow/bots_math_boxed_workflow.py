import fcntl
import json
import os
from typing import List, Union

from trinity.common.experience import Experience
from trinity.common.workflows.customized_math_workflows import MathBoxedWorkflow, Task
from trinity.common.workflows.workflow import WORKFLOWS


@WORKFLOWS.register_module("bots_math_boxed_workflow")
class BOTSMathBoxedWorkflow(MathBoxedWorkflow):
    """A workflow for math tasks that give answers in boxed format for BOTS."""

    def reset(self, task: Task):
        super().reset(task)
        from trinity.plugins.bots_math_boxed_reward import BOTSMathBoxedRewardFn

        self.reward_fn = BOTSMathBoxedRewardFn(**self.reward_fn_args)
        self.task_desc = nested_query(self.format_args.prompt_key, self.raw_task)
        self.truth = nested_query(self.format_args.response_key, self.raw_task)

    def format_messages(self):
        # the prompts are already in message format
        return self.task_desc


@WORKFLOWS.register_module("bots_ref_eval_collect_math_boxed_workflow")
class BOTSRefEvalCollectMathBoxedWorkflow(MathBoxedWorkflow):
    """A reference evaluation collection workflow for math tasks that give answers in boxed format for BOTS."""

    def reset(self, task: Task):
        super().reset(task)
        from trinity.plugins.bots_math_boxed_reward import BOTSMathBoxedRewardFn

        self.reward_fn = BOTSMathBoxedRewardFn(**self.reward_fn_args)
        self.task_desc = nested_query(self.format_args.prompt_key, self.raw_task)
        self.truth = nested_query(self.format_args.response_key, self.raw_task)

    def format_messages(self):
        # the prompts are already in message format
        return self.task_desc

    def run(self) -> List[Experience]:
        responses = super().run()

        rewards = [response.reward for response in responses]

        log_entry = {
            "model_version": self.model.model_version,
            "rewards": rewards,
            "question": self.task_desc,
            "truth": self.truth,
        }

        log_file_path = os.environ.get("BOTS_REF_EVAL_LOG_FILE", "./bots_ref_eval_log.jsonl")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        with open(log_file_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(log_entry, f)
                f.write("\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        return responses


def nested_query(query_key: str, query_obj: Union[dict, None]):
    # support nested query for a dict given query_keys split by '.'
    if query_obj is None:
        return None
    if "." in query_key:
        query_keys = query_key.split(".")
    else:
        query_keys = [query_key]
    ret = query_obj
    for key in query_keys:
        if isinstance(ret, dict) and key in ret:
            ret = ret[key]
        else:
            return None
    return ret
