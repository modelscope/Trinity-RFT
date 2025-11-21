# -*- coding: utf-8 -*-
"""Test for the workflow module"""
import asyncio
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
from unittest import mock
from unittest.mock import MagicMock

import openai
import ray
from parameterized import parameterized, parameterized_class
from torch import Tensor

from tests.common.vllm_test import CHAT_TEMPLATE
from tests.tools import get_model_path, get_template_config, get_unittest_dataset_config
from trinity.common.experience import EID, Experience
from trinity.common.models import create_inference_models
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards import RMGalleryFn
from trinity.common.workflows import (
    WORKFLOWS,
    MathBoxedWorkflow,
    MathEvalWorkflow,
    MathRMWorkflow,
    MathWorkflow,
    Workflow,
)
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task
from trinity.explorer.workflow_runner import WorkflowRunner


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
    can_reset: bool = True
    can_repeat: bool = True

    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]
        self.repeat_times = task.rollout_args.n
        if auxiliary_models is not None:
            for model in auxiliary_models:
                assert isinstance(model, openai.OpenAI)

    def reset(self, task: Task):
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    def run(self):
        exps = []
        if self.output_format == "json":
            import json

            for i in range(self.repeat_times):
                exp = Experience(tokens=Tensor([0, 1, 2, 3]), prompt_length=1)
                exp.response_text = json.dumps(self.obj)
                exps.append(exp)
            return exps
        elif self.output_format == "yaml":
            import yaml

            for i in range(self.repeat_times):
                exp = Experience(tokens=Tensor([0, 1, 2, 3]), prompt_length=1)
                exp.response_text = yaml.safe_dump(self.obj)
                exps.append(exp)
            return exps
        else:
            raise ValueError("Invalid output format")


class DummyAsyncWorkflow(Workflow):
    can_reset: bool = True
    can_repeat: bool = True
    is_async: bool = True

    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]
        self.repeat_times = task.rollout_args.n
        if auxiliary_models is not None:
            for model in auxiliary_models:
                assert isinstance(model, openai.AsyncOpenAI)

    def reset(self, task: Task):
        self.obj = task.raw_task
        self.output_format = task.workflow_args["output_format"]

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    async def run_async(self):
        await asyncio.sleep(0.1)
        exps = []
        if self.output_format == "json":
            import json

            for i in range(self.repeat_times):
                exp = Experience(tokens=Tensor([0, 1, 2, 3]), prompt_length=1)
                exp.response_text = json.dumps(self.obj)
                exps.append(exp)
            return exps
        elif self.output_format == "yaml":
            import yaml

            for i in range(self.repeat_times):
                exp = Experience(tokens=Tensor([0, 1, 2, 3]), prompt_length=1)
                exp.response_text = yaml.safe_dump(self.obj)
                exps.append(exp)
            return exps
        else:
            raise ValueError("Invalid output format")


class DummyMultiTurnWorkflow(MultiTurnWorkflow):
    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.contents = task.raw_task["contents"]  # type: ignore

    def run(self):
        memory = [{"role": "system", "content": "You are a helpful assistant."}]
        experience_list = []
        for content in self.contents:
            memory.append({"role": "user", "content": content})
            memory.append({"role": "assistant", "content": content.upper()})
            experience = self.process_messages_to_experience(memory, 0, {})
            experience_list.append(experience)
        return experience_list


class DummyAsyncMultiTurnWorkflow(MultiTurnWorkflow):
    is_async: bool = True

    def __init__(self, model, task: Task, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.contents = task.raw_task["contents"]  # type: ignore

    async def run_async(self):
        memory = [{"role": "system", "content": "You are a helpful assistant."}]
        experience_list = []
        for content in self.contents:
            await asyncio.sleep(0.1)
            memory.append({"role": "user", "content": content})
            memory.append({"role": "assistant", "content": content.upper()})
            experience = self.process_messages_to_experience(memory, 0, {})
            experience_list.append(experience)
        return experience_list


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
            repeat_times=taskset_config.repeat_times,
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
            repeat_times=taskset_config.repeat_times,
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
            repeat_times=taskset_config.repeat_times,
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
            repeat_times=taskset_config.repeat_times,
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
            repeat_times=taskset_config.repeat_times,
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
            repeat_times=taskset_config.repeat_times,
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
            repeat_times=taskset_config.repeat_times,
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
            repeat_times=taskset_config.repeat_times,
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
            repeat_times=taskset_config.repeat_times,
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

    @parameterized.expand([(DummyWorkflow,), (DummyAsyncWorkflow,)])
    def test_workflow_resettable(self, workflow_cls) -> None:
        model = MagicMock()
        json_task = Task(
            workflow=workflow_cls,
            repeat_times=1,
            raw_task={"a": 1},
            workflow_args={"output_format": "json"},
        )
        yaml_task = Task(
            workflow=workflow_cls,
            repeat_times=1,
            raw_task={"a": 1},
            workflow_args={"output_format": "yaml"},
        )
        workflow = json_task.to_workflow(model)
        if workflow.asynchronous:
            answer = asyncio.run(workflow.run_async())
        else:
            answer = workflow.run()
        self.assertEqual(answer[0].response_text, '{"a": 1}')
        workflow.reset(yaml_task)
        if workflow.asynchronous:
            answer = asyncio.run(workflow.run_async())
        else:
            answer = workflow.run()
        self.assertEqual(answer[0].response_text, "a: 1\n")

    @parameterized.expand([(DummyWorkflow,), (DummyAsyncWorkflow,)])
    def test_workflow_repeatable(self, workflow_cls) -> None:
        model = MagicMock()
        task = Task(
            workflow=workflow_cls,
            repeat_times=3,
            raw_task={"a": 1},
            workflow_args={"output_format": "json"},
        )
        workflow = task.to_workflow(model)
        workflow.set_repeat_times(2, run_id_base=0)
        self.assertEqual(workflow.repeat_times, 2)
        if workflow.asynchronous:
            answer = asyncio.run(workflow.run_async())
        else:
            answer = workflow.run()
        self.assertEqual(len(answer), 2)


@parameterized_class(
    ("workflow_cls",),
    [
        (DummyMultiTurnWorkflow,),
        (DummyAsyncMultiTurnWorkflow,),
    ],
)
class MultiTurnWorkflowTest(unittest.TestCase):
    def setUp(self):
        # configure the model
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_num = 1  # self.engine_num
        self.config.explorer.rollout_model.tensor_parallel_size = 1  # self.tensor_parallel_size
        self.config.explorer.rollout_model.chat_template = CHAT_TEMPLATE
        self.config.algorithm.repeat_times = 2  # self.repeat_times
        self.config.explorer.rollout_model.enable_history = True  # self.enable_history
        self.config.check_and_update()
        self.engines, self.auxiliary_engines = create_inference_models(self.config)
        self.model_wrapper = ModelWrapper(self.engines[0], engine_type="vllm", enable_history=True)

    def test_multi_turn_workflow(self):
        task = Task(
            workflow=self.workflow_cls,
            repeat_times=3,
            raw_task={"contents": ["hello world!", "how are you?"]},
            workflow_args={"output_format": "json"},
        )
        workflow = task.to_workflow(self.model_wrapper)
        workflow.set_repeat_times(2, run_id_base=0)
        if workflow.asynchronous:
            answer = asyncio.run(workflow.run_async())
        else:
            answer = workflow.run()
        self.assertEqual(len(answer), 2)

    def tearDown(self):
        ray.shutdown(_exiting_interpreter=True)


class StateRecordingWorkflow(Workflow):
    is_async: bool = True

    def __init__(self, *, task, model: ModelWrapper, auxiliary_models):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.wait_time = task.workflow_args.get("wait_time", 1)

    async def run_async(self):
        for i in range(self.wait_time):
            await self.model.set_workflow_state({"step": i})
            await asyncio.sleep(1)
        return [Experience(tokens=Tensor([0, 1, 2]), prompt_length=1, reward=1.0)]


class TestWorkflowStateRecording(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_state_recording(self):
        model = MagicMock()
        model_wrapper = ModelWrapper(model, engine_type="vllm")

        task = Task(
            workflow=StateRecordingWorkflow,
            repeat_times=3,
            raw_task={},
            workflow_args={"wait_time": 3},
        )
        workflow = task.to_workflow(model_wrapper)

        async def monitor_routine():
            old_state = {}
            count = 0
            for i in range(20):
                await asyncio.sleep(0.2)
                new_state = await model_wrapper.get_workflow_state()
                if new_state.get("step") != old_state.get("step"):
                    old_state = new_state
                    count += 1
            self.assertEqual(count, 3)
            return count

        await asyncio.gather(*[monitor_routine(), workflow.run_async()])


class TestAgentScopeWorkflowAdapter(unittest.IsolatedAsyncioTestCase):
    @unittest.skip("Waiting for agentscope>=0.1.6")
    async def test_adapter(self):
        try:
            from agentscope.model import TrinityChatModel
        except ImportError:
            self.skipTest("agentscope >= 0.1.6 is not installed")

        async def as_workflow_func(task, model) -> float:
            self.assertIsInstance(task, dict)
            self.assertIsInstance(model, TrinityChatModel)
            return task["reward"]

        model = MagicMock()
        openai_client = MagicMock()
        openai_client.model_path = "Qwen/Qwen3-8B"
        model.get_openai_async_client.return_value = openai_client
        model.extract_experience_from_history.return_value = [
            Experience(tokens=Tensor([0, 1, 2]), prompt_length=1, logprobs=Tensor([0.1, 0.2])),
            Experience(tokens=Tensor([3, 4, 5]), prompt_length=2, logprobs=Tensor([0.3])),
        ]

        as_adapter_cls = WORKFLOWS.get("agentscope_workflow_adapter")
        as_adapter = as_adapter_cls(
            task=Task(
                raw_task={"reward": 0.1},
                workflow_args={"workflow_func": as_workflow_func},
            ),
            model=model,
        )
        result = await as_adapter.run_async()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].reward, 0.1)
        self.assertEqual(result[0].prompt_length, 1)
        self.assertEqual(result[1].reward, 0.1)
        self.assertEqual(result[1].prompt_length, 2)


class DummyModelWrapper:
    def __init__(self, model, engine_type="vllm", **kwargs):
        pass

    async def prepare(self):
        return None

    def get_openai_client(self):
        return openai.OpenAI(api_key="EMPTY")

    def get_openai_async_client(self):
        return openai.AsyncOpenAI(api_key="EMPTY")

    async def clean_workflow_state(self):
        return

    @property
    async def model_version_async(self):
        return 0


class TestWorkflowRunner(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_runner(self):
        config = get_template_config()

        with mock.patch(
            "trinity.explorer.workflow_runner.ModelWrapper",
            DummyModelWrapper,
        ):
            runner = WorkflowRunner(
                config,
                model=MagicMock(),
                auxiliary_models=[MagicMock(), MagicMock()],
                runner_id=0,
            )
            await runner.prepare()

            task = Task(
                workflow=DummyWorkflow,
                repeat_times=3,
                raw_task={"a": 1},
                workflow_args={"output_format": "json"},
            )

            status, exps = await runner.run_task(task, repeat_times=3, run_id_base=0)

            self.assertTrue(status.ok)
            self.assertIsInstance(exps, list)
            self.assertEqual(len(exps), 3)

            task = Task(
                workflow=DummyAsyncWorkflow,
                repeat_times=2,
                raw_task={"a": 1},
                workflow_args={"output_format": "yaml"},
            )

            status, exps = await runner.run_task(task, repeat_times=2, run_id_base=0)
            self.assertTrue(status.ok)
            self.assertIsInstance(exps, list)
            self.assertEqual(len(exps), 2)

    async def test_workflow_runner_get_state(self):
        config = get_template_config()

        async def mock_get_api_server_url_remote():
            return None

        async def mock_get_model_version_remote():
            return 1

        model = MagicMock()
        model.get_api_server_url.remote = MagicMock(side_effect=mock_get_api_server_url_remote)
        model.get_model_version.remote = MagicMock(side_effect=mock_get_model_version_remote)

        runner = WorkflowRunner(
            config,
            model=model,
            auxiliary_models=[],
            runner_id=1,
        )
        await runner.prepare()

        task = Task(
            workflow=StateRecordingWorkflow,
            raw_task={},
            workflow_args={"wait_time": 2},
            batch_id=1,
            task_id=2,
        )

        async def monitor_routine():
            state_history = defaultdict(set)
            count = 0
            for i in range(20):
                await asyncio.sleep(0.4)
                new_state = await runner.get_runner_state()
                for k, v in new_state.items():
                    state_history[k].add(v)
            self.assertEqual(len(state_history["model_version"]), 1)
            self.assertEqual(len(state_history["workflow_id"]), 3)
            self.assertEqual(len(state_history["begin_time"]), 3)
            self.assertEqual(len(state_history["step"]), 2)
            return count

        await asyncio.gather(
            *[monitor_routine(), runner.run_task(task, repeat_times=3, run_id_base=0)]
        )
