"""SQLAlchemy models for different data."""

from typing import Dict, Optional, Tuple

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    create_engine,
    func,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from trinity.common.experience import Experience
from trinity.utils.log import get_logger

Base = declarative_base()


class TaskModel(Base):  # type: ignore
    """Model for storing tasks in SQLAlchemy."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_task = Column(JSON, nullable=False)

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(raw_task=dict)


class ExperienceModel(Base):  # type: ignore
    """SQLAlchemy model for Experience."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, server_default=func.now())
    task_id = Column(String(64), nullable=True, index=True)  # associated task id
    run_id = Column(Integer, nullable=True, index=True)  # associated run id
    msg_id = Column(String(64), nullable=True, index=True)  # associated message id
    # serialized experience object
    model_version = Column(Integer, nullable=True, index=True)
    experience_bytes = Column(LargeBinary, nullable=True)
    reward = Column(Float, nullable=True)
    consumed = Column(Integer, default=0, index=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        exp = Experience.deserialize(self.experience_bytes)
        exp.eid.task = self.task_id
        exp.eid.run = self.run_id
        exp.eid.suffix = self.msg_id
        exp.reward = self.reward
        exp.info["model_version"] = self.model_version
        return exp

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            reward=experience.reward,
            task_id=str(experience.eid.task),
            run_id=experience.eid.run,
            msg_id=str(experience.eid.suffix),
            model_version=experience.info["model_version"],
        )


class SFTDataModel(Base):  # type: ignore
    """SQLAlchemy model for SFT data."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_list = Column(JSON, nullable=True)
    experience_bytes = Column(LargeBinary, nullable=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.experience_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            message_list=experience.messages,
        )


class DPODataModel(Base):  # type: ignore
    """SQLAlchemy model for DPO data."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    chosen_message_list = Column(JSON, nullable=True)
    rejected_message_list = Column(JSON, nullable=True)
    experience_bytes = Column(LargeBinary, nullable=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.experience_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            chosen_message_list=experience.chosen_messages,
            rejected_message_list=experience.rejected_messages,
        )


def init_engine(db_url: str, table_name: str, schema_type: Optional[str]) -> Tuple:
    """Get the sqlalchemy engine."""
    logger = get_logger(__name__)
    engine = create_engine(db_url, poolclass=NullPool)

    if schema_type is None:
        schema_type = "task"

    from trinity.buffer.schema import SQL_SCHEMA

    base_class = SQL_SCHEMA.get(schema_type)

    table_attrs = {
        "__tablename__": table_name,
        "__abstract__": False,
        "__table_args__": {"keep_existing": True},
    }
    table_cls = type(table_name, (base_class,), table_attrs)

    try:
        Base.metadata.create_all(engine, checkfirst=True)
        logger.info(f"Created table {table_name} for schema type {schema_type}.")
    except OperationalError:
        logger.warning(f"Failed to create table {table_name}, assuming it already exists.")

    return engine, table_cls
