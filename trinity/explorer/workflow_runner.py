# -*- coding: utf-8 -*-
"""The Workflow Runner Module."""
import asyncio
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from trinity.buffer import get_buffer_reader
from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.common.models import get_debug_inference_model
from trinity.common.models.model import InferenceModel, ModelWrapper
from trinity.common.workflows import Task, Workflow
from trinity.utils.log import get_logger


@dataclass(frozen=True)
class Status:
    """Status of the task running result."""

    ok: bool
    metric: dict[str, float]
    message: Optional[str] = None


class WorkflowRunner:
    """A Ray remote actor to run the workflow and generate experiences."""

    def __init__(
        self,
        config: Config,
        model: InferenceModel,
        auxiliary_models: Optional[List[InferenceModel]] = None,
        runner_id: Optional[int] = None,
    ) -> None:
        self.logger = get_logger(f"{config.explorer.name}_runner_{runner_id}", in_ray_actor=True)
        self.config = config
        self.model = model
        self.model_wrapper = ModelWrapper(
            model,
            config.explorer.rollout_model.engine_type,
            enable_lora=config.explorer.rollout_model.enable_lora,
            enable_history=config.explorer.rollout_model.enable_history,
        )
        self.auxiliary_models = [
            ModelWrapper(
                model,
            )
            for model in (auxiliary_models or [])
        ]
        self.auxiliary_model_clients = []
        self.auxiliary_model_async_clients = []
        self.workflow_instance: Workflow = None
        self.runner_id = runner_id

    async def prepare(self) -> None:
        """Prepare the runner."""
        await asyncio.gather(
            self.model_wrapper.prepare(),
            *(aux_model.prepare() for aux_model in self.auxiliary_models),
        )
        for model in self.auxiliary_models:
            api_client = model.get_openai_client()
            async_api_client = model.get_openai_async_client()
            self.auxiliary_model_clients.append(api_client)
            self.auxiliary_model_async_clients.append(async_api_client)

    def is_alive(self):
        return True

    def _create_workflow_instance(self, task: Task) -> None:
        if task.workflow is None:
            raise ValueError("Workflow is not set in the task.")
        if (
            self.workflow_instance is None
            or not self.workflow_instance.__class__ == task.workflow
            or not self.workflow_instance.resettable
        ):
            self.workflow_instance = task.to_workflow(
                self.model_wrapper,
                (
                    self.auxiliary_model_async_clients
                    if task.workflow.is_async
                    else self.auxiliary_model_clients
                ),
            )
        else:
            self.workflow_instance.reset(task)

    async def _run_workflow(self, workflow_instance: Workflow) -> List[Experience]:
        if workflow_instance.asynchronous:
            exps = await workflow_instance.run_async()
        else:
            exps = workflow_instance.run()
        return exps

    async def _run_task(self, task: Task, repeat_times: int, run_id_base: int) -> List[Experience]:
        """Init workflow from the task and run it."""
        self._create_workflow_instance(task)
        if self.workflow_instance.repeatable:
            self.workflow_instance.set_repeat_times(repeat_times, run_id_base)
            exps = await self._run_workflow(self.workflow_instance)
        else:
            exps = []
            for i in range(repeat_times):
                new_exps = await self._run_workflow(self.workflow_instance)
                for exp in new_exps:
                    exp.eid.run = run_id_base + i
                exps.extend(new_exps)
                if i < repeat_times - 1:
                    self._create_workflow_instance(task)
        return exps

    async def run_task(
        self,
        task: Task,
        repeat_times: int = 1,
        run_id_base: int = 0,
    ) -> Tuple[Status, List[Experience]]:
        """Run the task and return the states."""
        # TODO: avoid sending the experiences back to the scheduler to reduce the communication overhead
        try:
            st = time.time()
            exps = await self._run_task(task, repeat_times, run_id_base)
            assert exps is not None and len(exps) > 0, "An empty experience is generated"
            metrics: dict[str, List[float]] = defaultdict(list)
            model_version = await self.model_wrapper.model_version_async
            # set eid for each experience
            for i, exp in enumerate(exps):
                exp.eid.batch = task.batch_id
                # keep exp.eid.task if it has been set before (e.g., in workflow)
                if exp.eid.task == "":  # "" is the default value
                    exp.eid.task = task.task_id
                if not hasattr(exp, "info") or exp.info is None:
                    exp.info = {}
                exp.info["model_version"] = model_version
                exp.info["use_count"] = 0
                exp.info["task_index"] = task.index

                if not hasattr(exp, "metrics") or exp.metrics is None:
                    exp.metrics = {}
                for k, v in exp.metrics.items():
                    metrics[k].append(v)
            # We get the average of metrics into the state
            metric = {}
            metric["time_per_task"] = time.time() - st
            if metrics:
                for k, v in metrics.items():
                    metric[k] = sum(v) / len(v)  # type: ignore

            if task.is_eval:
                # If the task is an evaluation task, we do not record the experiences to the buffer
                return Status(True, metric=metric), []
            else:
                return Status(True, metric=metric), exps

        except Exception as e:
            error_trace_back = traceback.format_exc()
            self.logger.error(f"WorkflowRunner run task error: {e}\nTraceback:\n{error_trace_back}")
            return Status(False, metric={"time_per_task": time.time() - st}, message=str(e)), []


class DebugWorkflowRunner(WorkflowRunner):
    """A WorkflowRunner for debugging."""

    def __init__(
        self,
        config: Config,
        output_file: str,
    ) -> None:
        model, auxiliary_models = get_debug_inference_model(config)
        super().__init__(config, model, auxiliary_models, 0)
        self.taskset = get_buffer_reader(config.buffer.explorer_input.tasksets[0])
        self.output_file = output_file

    async def debug(self) -> None:
        """Run the debug workflow."""
        from viztracer import VizTracer

        await self.prepare()
        tasks = await self.taskset.read_async(batch_size=1)
        task = tasks[0]
        self.logger.info(f"Read task: {task.task_id}, repeat_times: {task.repeat_times}")
        with VizTracer(output_file=self.output_file):
            status, exps = await self.run_task(task, task.repeat_times, 0)
        if status.ok:
            print(f"Task {task.task_id} completed successfully with metrics:\n{status.metric}")
            for exp in exps:
                print(f"Generated experience:\n{exp}")
        else:
            self.logger.error(f"Task {task.task_id} failed with message: {status.message}")
        self.logger.info("Debugging completed.")
