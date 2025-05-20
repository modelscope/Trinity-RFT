# -*- coding: utf-8 -*-
"""The Workflow Runner Moudle."""
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import ray

from trinity.buffer import get_buffer_writer
from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel, ModelWrapper
from trinity.common.workflows import Task
from trinity.utils.log import get_logger


@dataclass(frozen=True)
class Status:
    """Status of the task running result."""

    ok: bool
    metric: dict[str, float]
    message: Optional[str] = None


@ray.remote(scheduling_strategy="SPREAD")
class WorkflowRunner:
    """A Ray remote actor to run the workflow and put the returned experiences into the buffer."""

    def __init__(self, config: Config, model: InferenceModel) -> None:
        self.config = config
        self.experience_buffer = get_buffer_writer(
            self.config.buffer.explorer_output,  # type: ignore
            self.config.buffer,
        )
        self.model = model
        self.model_wrapper = ModelWrapper(model, config.explorer.engine_type)
        self.logger = get_logger(__name__)
        self.workflow_instance = None

    def is_alive(self):
        return True

    def _run_task(self, task: Task) -> List[Experience]:
        """Init workflow from the task and run it."""
        if task.workflow is None:
            raise ValueError("Workflow is not set in the task.")
        if (
            self.workflow_instance is None
            or not self.workflow_instance.__class__ == task.workflow
            or not self.workflow_instance.resettable
        ):
            self.workflow_instance = task.to_workflow(self.model_wrapper)
        else:
            self.workflow_instance.reset(task)
        return self.workflow_instance.run()

    def run_task(self, task: Task) -> Status:
        """Run the task and return the states."""
        try:
            st = time.time()
            exps = self._run_task(task)
            assert exps is not None and len(exps) > 0, "An empty experience is generated"
            metrics: dict[str, List[float]] = defaultdict(list)
            # add run_id
            run_id = str(uuid.uuid4())
            for exp in exps:
                setattr(exp, "run_id", run_id)

                if not hasattr(exp, "info") or exp.info is None:
                    exp.info = {}
                exp.info["model_version"] = self.model_wrapper.get_ckp_version()

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
            if not task.is_eval:
                self.experience_buffer.write(exps)
            return Status(True, metric=metric)
        except Exception as e:
            error_trace_back = traceback.format_exc()
            self.logger.error(f"WorkflowRunner run task error: {e}\nTraceback:\n{error_trace_back}")
            return Status(False, metric={"time_per_task": time.time() - st}, message=str(e))
