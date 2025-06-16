import time
from typing import List, Optional

import ray
from sqlalchemy import asc, create_engine, desc
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from trinity.buffer.schema import Base, create_dynamic_table
from trinity.buffer.utils import retry_session
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import ReadStrategy
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

    @classmethod
    def get_wrapper(cls, storage_config: StorageConfig, config: BufferConfig):
        if storage_config.wrap_in_ray:
            return (
                ray.remote(cls)
                .options(
                    name=f"sql-{storage_config.name}",
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
