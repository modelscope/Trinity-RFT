"""Reader of the SQL buffer."""

import time
from typing import List, Optional

from sqlalchemy import asc, create_engine, desc
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.schema import Base, create_dynamic_table
from trinity.buffer.utils import retry_session
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import ReadStrategy, StorageType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class SQLReader(BufferReader):
    """Reader of the SQL buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig) -> None:
        assert meta.storage_type == StorageType.SQL
        self.engine = create_engine(meta.path, poolclass=NullPool)

        self.table_model_cls = create_dynamic_table(meta.algorithm_type, meta.name)
        try:
            Base.metadata.create_all(self.engine, checkfirst=True)
        except OperationalError:
            logger.warning("Failed to create database, assuming it already exists.")
        self.session = sessionmaker(bind=self.engine)
        self.batch_size = config.read_batch_size
        self.max_retry_times = config.max_retry_times
        self.max_retry_interval = config.max_retry_interval

    def read(self, strategy: Optional[ReadStrategy] = None) -> List:
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
        while len(exp_list) < self.batch_size:
            if len(exp_list):
                logger.info("waiting for experiences...")
                time.sleep(1)
            with retry_session(
                self.session, self.max_retry_times, self.max_retry_interval
            ) as session:
                # get a batch of experiences from the database
                experiences = (
                    session.query(self.table_model_cls)
                    .filter(self.table_model_cls.reward.isnot(None))
                    .order_by(*sortOrder)  # TODO: very slow
                    .limit(self.batch_size - len(exp_list))
                    .with_for_update()
                    .all()
                )
                # update the consumed field
                for exp in experiences:
                    exp.consumed += 1
                exp_list.extend([self.table_model_cls.to_experience(exp) for exp in experiences])
        logger.info(f"get {len(exp_list)} experiences:")
        logger.info(f"reward = {[exp.reward for exp in exp_list]}")
        logger.info(f"first prompt_text = {exp_list[0].prompt_text}")
        logger.info(f"first response_text = {exp_list[0].response_text}")
        return exp_list
