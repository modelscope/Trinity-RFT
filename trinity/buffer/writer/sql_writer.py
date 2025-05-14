"""Writer of the SQL buffer."""

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.schema import Base, create_dynamic_table
from trinity.buffer.utils import retry_session
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class SQLWriter(BufferWriter):
    """Writer of the SQL buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig) -> None:
        assert meta.storage_type == StorageType.SQL
        # we only support write RFT algorithm buffer for now
        # TODO: support other algorithms
        assert meta.algorithm_type.is_rft, "Only RFT buffer is supported for writing."
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

    def write(self, data: list) -> None:
        with retry_session(self.session, self.max_retry_times, self.max_retry_interval) as session:
            experience_models = [self.table_model_cls.from_experience(exp) for exp in data]
            session.add_all(experience_models)

    def finish(self) -> None:
        # TODO: implement this
        pass
