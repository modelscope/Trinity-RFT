# -*- coding: utf-8 -*-
"""Test for the workflow module"""
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

from tests.tools import get_unittest_dataset_config
from trinity.common.workflows import MathWorkflow
from trinity.common.workflows.workflow import Task


@dataclass
class MockResponse:
    response_text: str
    reward: float = 0.0


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

    def test_gsm8k_workflow(self) -> None:
        model = MagicMock()
        model.chat.return_value = [
            MockResponse("<think> balabalabala 99 </think>\n<answer> 36 </answer>"),
            MockResponse("<answer> 36.0 </answer>"),
            MockResponse("<answer>Kim's total points are 6 + 30 = 36 </answer>"),
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
        # self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0].reward, 1.1)
        self.assertEqual(experiences[1].reward, 0.9)
        self.assertEqual(experiences[2].reward, 0.9)
