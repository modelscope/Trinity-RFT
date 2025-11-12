from typing import Union

from trinity.common.workflows.customized_math_workflows import MathBoxedWorkflow, Task
from trinity.common.workflows.workflow import WORKFLOWS

from .bots_math_boxed_reward import BOTSMathBoxedRewardFn


@WORKFLOWS.register_module("bots_math_boxed_workflow")
class BOTSMathBoxedWorkflow(MathBoxedWorkflow):
    """A workflow for math tasks that give answers in boxed format for BOTS."""

    def reset(self, task: Task):
        super().reset(task)
        self.reward_fn = BOTSMathBoxedRewardFn(**self.reward_fn_args)

    def format_messages(self):
        # the prompts are already in message format
        return self.task_desc

    @property
    def task_desc(self) -> Union[str, None]:  # type: ignore [override]
        prompt_key = self.format_args.prompt_key
        return nested_query(prompt_key, self.raw_task)  # type: ignore

    @property
    def truth(self) -> Union[str, None]:  # type: ignore [override]
        response_key = self.format_args.response_key
        return nested_query(response_key, self.raw_task)


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
