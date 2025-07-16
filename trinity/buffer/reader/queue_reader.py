"""Reader of the Queue buffer."""

from typing import List, Optional

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.queue import QueueActor
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import ReadStrategy, StorageType
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

logger = get_logger(__name__)

RETURN_TYPES = Registry("return_types")

@RETURN_TYPES.register_module("experience")
class QueueReader(BufferReader):
    """Reader of the Queue buffer."""

    def __init__(self, storage_config: StorageConfig, config: BufferConfig):
        assert storage_config.storage_type == StorageType.QUEUE
        self.timeout = storage_config.max_read_timeout
        self.read_batch_size = config.read_batch_size
        self.queue = QueueActor.get_actor(storage_config, config)

    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        if strategy is not None and strategy != ReadStrategy.FIFO:
            raise NotImplementedError(f"Read strategy {strategy} not supported for Queue Reader.")
        try:
            batch_size = batch_size or self.read_batch_size
            exps = ray.get(self.queue.get_batch.remote(batch_size, timeout=self.timeout))
            if len(exps) != batch_size:
                raise TimeoutError(
                    f"Read incomplete batch ({len(exps)}/{batch_size}), please check your workflow."
                )
        except StopAsyncIteration:
            raise StopIteration()
        return exps

@RETURN_TYPES.register_module("rollout")
class QueueRolloutDataReader(QueueReader):

    def __init__(self, storage_config: StorageConfig, config: BufferConfig):
        super().__init__(storage_config, config)
        self.storage_config = storage_config
        self.workflow_key = storage_config.format.workflow_key
        self.reward_fn_key = storage_config.format.reward_fn_key
        self.default_workflow_cls = WORKFLOWS.get(meta.default_workflow_type)  # type: ignore
        self.default_reward_fn_cls = REWARD_FUNCTIONS.get(meta.default_reward_fn_type)  # type: ignore

    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        from trinity.common.rewards import REWARD_FUNCTIONS
        from trinity.common.workflows import WORKFLOWS, Task
        from trinity.common.constants import TaskType

        batch_size = batch_size or self.read_batch_size
        samples = super().read(batch_size, strategy)
        tasks = []
        for sample in samples:
            workflow_class = (
                WORKFLOWS.get(sample[self.workflow_key])
                if self.workflow_key in sample
                else self.default_workflow_cls
            )
            reward_fn = (
                REWARD_FUNCTIONS.get(sample[self.reward_fn_key])
                if self.reward_fn_key in sample
                else self.default_reward_fn_cls
            )
            assert (
                workflow_class is not None
            ), "`default_workflow_type` or `workflow_key` is required"
            task = Task(
                workflow=workflow_class,
                format_args=self.storage_config.format,
                rollout_args=self.storage_config.rollout_args,
                workflow_args=self.storage_config.workflow_args,
                reward_fn_args=self.storage_config.reward_fn_args,
                is_eval=self.storage_config.task_type == TaskType.EVAL,
                reward_fn=reward_fn,
                raw_task=sample,
            )
            tasks.append(task)
        return tasks
