# -*- coding: utf-8 -*-
"""Test for the workflow module"""
import unittest
from dataclasses import dataclass, field
from typing import Dict, Optional
from unittest.mock import MagicMock

from torch import Tensor

from tests.tools import get_unittest_dataset_config
from trinity.common.experience import EID
from trinity.common.rewards import RMGalleryFn
from trinity.common.workflows import (
    MathBoxedWorkflow,
    MathEvalWorkflow,
    MathRMWorkflow,
    MathWorkflow,
    Workflow,
)
from trinity.common.workflows.workflow import Task


@dataclass
class MockResponse:
    response_text: str
    reward: float = 0.0
    metrics: Optional[Dict[str, float]] = None
    info: Optional[Dict] = None
    unique_id: Optional[str] = "0"
    tokens: Optional[Tensor] = Tensor([0, 0])
    prompt_length: int = 1
    eid: EID = field(default_factory=EID)


class DummyWorkflow(Workflow):
    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]

    @property
    def resettable(self):
        return True

    def reset(self, task: Task):
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]

    def run(self):
        if self.output_format == "json":
            import json

            return [json.dumps(self.obj)]
        elif self.output_format == "yaml":
            import yaml

            return [yaml.safe_dump(self.obj)]
        else:
            raise ValueError("Invalid output format")


class WorkflowTest(unittest.TestCase):
    def test_math_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse(r"\boxed{2}"),
            MockResponse(r"\boxted{3}"),
            MockResponse(r"2"),
            MockResponse("<think>\nThinking\n</think>\n<answer>\n3\n</answer>"),
            MockResponse("<think>\nThinking\n</think>\n<answer>\n\\boxed{2}\n</answer>"),
            MockResponse("<think>Missing closing</think><answer>\\boxed{2}"),
            MockResponse("<answer>\nOnly answer\n</answer>"),
            MockResponse("<think>\nOnly thinking\n</think>"),
            MockResponse("<think>Thinking</think><answer>Answer is not end</answer><answer>1"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathWorkflow,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "1+1=",
                taskset_config.format.response_key: "2",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(len(experiences), 9)
        self.assertEqual(experiences[0].reward, 0.9)
        self.assertEqual(experiences[1].reward, -0.1)
        self.assertEqual(experiences[2].reward, 0.9)
        self.assertEqual(experiences[3].reward, 0.1)
        self.assertEqual(experiences[4].reward, 1.1)
        self.assertEqual(experiences[5].reward, 0.9)
        self.assertEqual(experiences[6].reward, -0.1)
        self.assertEqual(experiences[7].reward, -0.1)
        self.assertEqual(experiences[8].reward, -0.1)

    def test_math_fraction_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse(r"\boxed{\frac{40}{400}}"),
            MockResponse(r"\boxed{\frac{1}{10}}"),
            MockResponse(r"\boxed{0.1}"),
            MockResponse(r"\boxed{0.1000}"),
            MockResponse(r"\boxed{\frac{1} {10}}"),
            MockResponse(r"The answer is \boxed{\frac{40}{400}}"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathWorkflow,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: r"\frac{40}{400}",
                taskset_config.format.response_key: r"\frac{40}{400}",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(len(experiences), 6)
        self.assertEqual(experiences[0].reward, 0.9)
        self.assertEqual(experiences[1].reward, 0.9)
        self.assertEqual(experiences[2].reward, 0.9)
        self.assertEqual(experiences[3].reward, 0.9)
        self.assertEqual(experiences[4].reward, 0.9)
        self.assertEqual(experiences[5].reward, 0.9)

    def test_math_complex_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse(
                r"$\boxed{\dfrac{108 + 31\sqrt{5}}{216}} \quad \text{and} \quad \boxed{\dfrac{108 - 31\sqrt{5}}{216}}$"
            ),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathWorkflow,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"$x_{1}=\frac{1}{2}+\frac{31\sqrt{5}}{216},\quadx_{2}=\frac{1}{2}-\frac{31\sqrt{5}}{216}$",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0].reward, 0.9)

    def test_math_boxed_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse("<think> balabalabala 99 </think>\n \\boxed{36}"),
            MockResponse("answer is \\boxed{36 }"),
            MockResponse("Kim's total points are 6 + 30 =\\boxed{36}"),
            MockResponse("<think> balalaba </think> \\boxed{35.00}"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathBoxedWorkflow,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            workflow_args={
                "with_think": False,
                "format_score_coef": 0.2,
            },
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"36",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(experiences[0].reward, 1.0)
        self.assertEqual(experiences[1].reward, 1.0)
        self.assertEqual(experiences[2].reward, 1.0)
        self.assertEqual(experiences[3].reward, 0.0)
        task_new = Task(
            workflow=MathBoxedWorkflow,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            workflow_args={
                "with_think": True,
                "format_score_coef": 0.2,
            },
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"36",
            },
        )
        workflow.reset(task_new)
        workflow_new = task_new.to_workflow(model=model)
        experiences = workflow_new.run()
        self.assertEqual(experiences[0].reward, 1.0)
        self.assertEqual(experiences[1].reward, 0.8)
        self.assertEqual(experiences[2].reward, 0.8)
        self.assertEqual(experiences[3].reward, 0.0)

    def test_gsm8k_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse("<think> balabalabala 99 </think>\n<answer> 36 </answer>"),
            MockResponse("<answer> 36.0 </answer>"),
            MockResponse("<answer>Kim's total points are 6 + 30 = 36 </answer>"),
            MockResponse("<think> balalaba </think><answer> 35.00 </answer>"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathWorkflow,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"36",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(experiences[0].reward, 1.1)
        self.assertEqual(experiences[1].reward, 0.9)
        self.assertEqual(experiences[2].reward, 0.9)
        self.assertEqual(experiences[3].reward, 0.1)
        task_new = Task(
            workflow=MathWorkflow,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"35",
            },
        )
        workflow.reset(task_new)
        workflow_new = task_new.to_workflow(model=model)
        experiences = workflow_new.run()
        self.assertEqual(experiences[0].reward, 0.1)
        self.assertEqual(experiences[1].reward, -0.1)
        self.assertEqual(experiences[2].reward, -0.1)
        self.assertEqual(experiences[3].reward, 1.1)

    def test_rm_gallery_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse("<think> balabalabala 99 </think>\n \\boxed{36}"),
            MockResponse("answer is \\boxed{36 }"),
            MockResponse("Kim's total points are 6 + 30 =\\boxed{36}"),
            MockResponse("<think> balalaba </think> \\boxed{35.00}"),
        ]
        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathRMWorkflow,
            reward_fn=RMGalleryFn,
            format_args=taskset_config.format,
            rollout_args=taskset_config.rollout_args,
            reward_fn_args={
                "reward_name": "math_verify_reward",
            },
            is_eval=False,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: r"36",
            },
        )
        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(experiences[0].reward, 1.0)
        self.assertEqual(experiences[1].reward, 1.0)
        self.assertEqual(experiences[2].reward, 1.0)
        self.assertEqual(experiences[3].reward, 0.0)

    def test_math_eval_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse("My step-by-step reasoning leads to the answer \boxed{36}"),
            MockResponse("Here is the answer of \boxed{36.0}"),
            MockResponse("I made a mistake, the answer is \boxed{42}"),
            MockResponse("The answer is 36, but I forgot the box."),
        ]

        taskset_config = get_unittest_dataset_config("countdown")
        task = Task(
            workflow=MathEvalWorkflow,
            is_eval=True,
            format_args=taskset_config.format,
            raw_task={
                taskset_config.format.prompt_key: "",
                taskset_config.format.response_key: "36",
            },
        )

        workflow = task.to_workflow(model=model)
        experiences = workflow.run()
        self.assertEqual(len(experiences), 4)
        expected_accuracies = [1.0, 1.0, 0.0, 0.0]
        for i, (exp, expected_acc) in enumerate(zip(experiences, expected_accuracies)):
            with self.subTest(f"Response {i}"):
                self.assertEqual(exp.reward, 0.0)
                assert exp.metrics is not None, f"Metrics for response {i} should not be None"
                self.assertEqual(exp.metrics["accuracy"], expected_acc)

    def test_workflow_resettable(self) -> None:
        model = MagicMock()
        json_task = Task(
            workflow=DummyWorkflow, raw_task={"a": 1}, workflow_args={"output_format": "json"}
        )
        yaml_task = Task(
            workflow=DummyWorkflow, raw_task={"a": 1}, workflow_args={"output_format": "yaml"}
        )
        workflow = json_task.to_workflow(model)
        answer = workflow.run()
        self.assertEqual(answer[0], '{"a": 1}')
        workflow.reset(yaml_task)
        answer = workflow.run()
        self.assertEqual(answer[0], "a: 1\n")
