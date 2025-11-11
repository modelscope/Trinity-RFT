from typing import List

from trinity.common.workflows import WORKFLOWS, Workflow
from trinity.common.workflows.workflow import MathWorkflow


@WORKFLOWS.register_module("my_workflow")
class MyWorkflow(Workflow):
    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)

    @property
    def repeatable(self):
        return True

    def set_repeat_times(self, repeat_times, run_id_base):
        pass

    def run(self) -> List:
        return ["Hello world", "Hi"]


@WORKFLOWS.register_module("custom_workflow")
class CustomWorkflow(MathWorkflow):
    def run(self):
        responses = super().run()
        for i, response in enumerate(responses):
            response.metrics["custom_metric"] = i
        return responses
