from typing import Union

from examples.bots.workflow.bots_math_boxed_reward import BOTSMathBoxedRewardFn
from trinity.common.workflows.customized_math_workflows import MathBoxedWorkflow, Task


class BOTSMathBoxedWorkflow(MathBoxedWorkflow):
    """A workflow for math tasks that give answers in boxed format for BOTS."""

    def reset(self, task: Task):
        super().reset(task)

        self.reward_fn = BOTSMathBoxedRewardFn(**self.reward_fn_args)
        self.task_desc = nested_query(self.format_args.prompt_key, self.raw_task)
        self.truth = nested_query(self.format_args.response_key, self.raw_task)

    def format_messages(self):
        # the prompts are already in message format
        return self.task_desc


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
