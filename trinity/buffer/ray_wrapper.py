"""Ray actor wrapper for different buffers."""
import asyncio
import json
import os
import time
from collections import deque
from copy import deepcopy
from typing import List, Optional

import ray
from sqlalchemy import asc, create_engine, desc
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from trinity.buffer.queue import QueueBuffer
from trinity.buffer.schema import Base, create_dynamic_table
from trinity.buffer.utils import default_storage_path, retry_session
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import ReadStrategy, StorageType
from trinity.common.experience import EID, Experience
from trinity.common.workflows import Task
from trinity.utils.log import get_logger


class DBWrapper:
    """
    A wrapper of a SQL database.

    If `wrap_in_ray` in `StorageConfig` is `True`, this class will be run as a Ray Actor,
    and provide a remote interface to the local database.

    For databases that do not support multi-processing read/write (e.g. sqlite, duckdb), we
    recommend setting `wrap_in_ray` to `True`
    """

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        self.logger = get_logger(__name__)
        if storage_config.path is None:
            storage_config.path = default_storage_path(storage_config, config)
        self.engine = create_engine(storage_config.path, poolclass=NullPool)
        self.table_model_cls = create_dynamic_table(
            storage_config.algorithm_type, storage_config.name
        )

        try:
            Base.metadata.create_all(self.engine, checkfirst=True)
        except OperationalError:
            self.logger.warning("Failed to create database, assuming it already exists.")

        self.session = sessionmaker(bind=self.engine)
        self.batch_size = config.read_batch_size
        self.max_retry_times = config.max_retry_times
        self.max_retry_interval = config.max_retry_interval
        self.ref_count = 0
        self.stopped = False

    @classmethod
    def get_wrapper(cls, storage_config: StorageConfig, config: BufferConfig):
        if storage_config.wrap_in_ray:
            return (
                ray.remote(cls)
                .options(
                    name=f"sql-{storage_config.name}",
                    namespace=storage_config.ray_namespace or ray.get_runtime_context().namespace,
                    get_if_exists=True,
                )
                .remote(storage_config, config)
            )
        else:
            return cls(storage_config, config)

    def write(self, data: list) -> None:
        with retry_session(self.session, self.max_retry_times, self.max_retry_interval) as session:
            experience_models = [self.table_model_cls.from_experience(exp) for exp in data]
            session.add_all(experience_models)

    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        if self.stopped:
            raise StopIteration()

        if strategy is None:
            strategy = ReadStrategy.LFU

        if strategy == ReadStrategy.LFU:
            sortOrder = (asc(self.table_model_cls.consumed), desc(self.table_model_cls.id))

        elif strategy == ReadStrategy.LRU:
            sortOrder = (desc(self.table_model_cls.id),)

        elif strategy == ReadStrategy.PRIORITY:
            sortOrder = (desc(self.table_model_cls.priority), desc(self.table_model_cls.id))

        else:
            raise NotImplementedError(f"Unsupported strategy {strategy} by SQLStorage")

        exp_list = []
        batch_size = batch_size or self.batch_size
        while len(exp_list) < batch_size:
            if len(exp_list):
                self.logger.info("waiting for experiences...")
                time.sleep(1)
            with retry_session(
                self.session, self.max_retry_times, self.max_retry_interval
            ) as session:
                # get a batch of experiences from the database
                experiences = (
                    session.query(self.table_model_cls)
                    .filter(self.table_model_cls.reward.isnot(None))
                    .order_by(*sortOrder)  # TODO: very slow
                    .limit(batch_size - len(exp_list))
                    .with_for_update()
                    .all()
                )
                # update the consumed field
                for exp in experiences:
                    exp.consumed += 1
                exp_list.extend([self.table_model_cls.to_experience(exp) for exp in experiences])
        self.logger.info(f"get {len(exp_list)} experiences:")
        self.logger.info(f"reward = {[exp.reward for exp in exp_list]}")
        self.logger.info(f"first prompt_text = {exp_list[0].prompt_text}")
        self.logger.info(f"first response_text = {exp_list[0].response_text}")
        return exp_list

    def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    def release(self) -> int:
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.stopped = True
        return self.ref_count


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Experience):
            return o.to_dict()
        if isinstance(o, Task):
            return o.to_dict()
        if isinstance(o, EID):
            return o.to_dict()
        return super().default(o)


class FileWrapper:
    """
    A wrapper of a local jsonl file.

    If `wrap_in_ray` in `StorageConfig` is `True`, this class will be run as
    a Ray Actor, and provide a remote interface to the local file.

    This wrapper is only for writing, if you want to read from the file, use
    StorageType.QUEUE instead.
    """

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        if storage_config.path is None:
            storage_config.path = default_storage_path(storage_config, config)
        ext = os.path.splitext(storage_config.path)[-1]
        if ext != ".jsonl" and ext != ".json":
            raise ValueError(
                f"File path must end with '.json' or '.jsonl', got {storage_config.path}"
            )
        path_dir = os.path.dirname(storage_config.path)
        os.makedirs(path_dir, exist_ok=True)
        self.file = open(storage_config.path, "a", encoding="utf-8")
        self.encoder = _Encoder(ensure_ascii=False)
        self.ref_count = 0

    @classmethod
    def get_wrapper(cls, storage_config: StorageConfig, config: BufferConfig):
        if storage_config.wrap_in_ray:
            return (
                ray.remote(cls)
                .options(
                    name=f"json-{storage_config.name}",
                    namespace=storage_config.ray_namespace or ray.get_runtime_context().namespace,
                    get_if_exists=True,
                )
                .remote(storage_config, config)
            )
        else:
            return cls(storage_config, config)

    def write(self, data: List) -> None:
        for item in data:
            json_str = self.encoder.encode(item)
            self.file.write(json_str + "\n")
        self.file.flush()

    def read(self) -> List:
        raise NotImplementedError(
            "read() is not implemented for FileWrapper, please use QUEUE instead"
        )

    def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    def release(self) -> int:
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.file.close()
        return self.ref_count


def is_database_url(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in ["sqlite:///", "postgresql://", "mysql://"])


def is_json_file(path: str) -> bool:
    return path.endswith(".json") or path.endswith(".jsonl")


class QueueWrapper:
    """An wrapper of a async queue."""

    def __init__(self, storage_config: StorageConfig, config: BufferConfig) -> None:
        self.logger = get_logger(__name__)
        self.config = config
        self.capacity = storage_config.capacity
        self.queue = QueueBuffer.get_queue(storage_config, config)
        st_config = deepcopy(storage_config)
        st_config.wrap_in_ray = False
        if st_config.path is not None:
            if is_database_url(st_config.path):
                from trinity.buffer.writer.sql_writer import SQLWriter

                st_config.storage_type = StorageType.SQL
                self.writer = SQLWriter(st_config, self.config)
            elif is_json_file(st_config.path):
                from trinity.buffer.writer.file_writer import JSONWriter

                st_config.storage_type = StorageType.FILE
                self.writer = JSONWriter(st_config, self.config)
            else:
                self.logger.warning("Unknown supported storage path: %s", st_config.path)
                self.writer = None
        else:
            from trinity.buffer.writer.file_writer import JSONWriter

            st_config.storage_type = StorageType.FILE
            self.writer = JSONWriter(st_config, self.config)
        self.logger.warning(f"Save experiences in {st_config.path}.")
        self.ref_count = 0
        self.exp_pool = deque()  # A pool to store experiences
        self.closed = False

    async def acquire(self) -> int:
        self.ref_count += 1
        return self.ref_count

    async def release(self) -> int:
        """Release the queue."""
        self.ref_count -= 1
        if self.ref_count <= 0:
            await self.queue.close()
            await self.writer.release()
        return self.ref_count

    def length(self) -> int:
        """The length of the queue."""
        return self.queue.qsize()

    async def put_batch(self, exp_list: List) -> None:
        """Put batch of experience."""
        await self.queue.put(exp_list)
        if self.writer is not None:
            self.writer.write(exp_list)

    async def get_batch(self, batch_size: int, timeout: float) -> List:
        """Get batch of experience."""
        start_time = time.time()
        while len(self.exp_pool) < batch_size:
            if self.queue.stopped():
                # If the queue is stopped, ignore the rest of the experiences in the pool
                raise StopAsyncIteration("Queue is closed and no more items to get.")
            try:
                exp_list = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.exp_pool.extend(exp_list)
            except asyncio.TimeoutError:
                if time.time() - start_time > timeout:
                    self.logger.error(
                        f"Timeout when waiting for experience, only get {len(self.exp_pool)} experiences.\n"
                        "This phenomenon is usually caused by the workflow not returning enough "
                        "experiences or running timeout. Please check your workflow implementation."
                    )
                    batch = list(self.exp_pool)
                    self.exp_pool.clear()
                    return batch
        return [self.exp_pool.popleft() for _ in range(batch_size)]

    @classmethod
    def get_wrapper(cls, storage_config: StorageConfig, config: BufferConfig):
        """Get the queue actor."""
        return (
            ray.remote(cls)
            .options(
                name=f"queue-{storage_config.name}",
                namespace=storage_config.ray_namespace or ray.get_runtime_context().namespace,
                get_if_exists=True,
            )
            .remote(storage_config, config)
        )
