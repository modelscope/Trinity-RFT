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
