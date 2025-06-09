"""Schema for SQLAlchemy models."""

from typing import Any, Optional, Union

from sqlalchemy import Column, Float, Integer, LargeBinary, String
from sqlalchemy.ext.declarative import declarative_base

from trinity.algorithm.algorithm import Algorithm
from trinity.common.experience import Experience
from trinity.common.models.utils import tokenize_and_mask_messages_hf

Base = declarative_base()


class TaskModel(Base):  # type: ignore
    """Model for storing tasks in SQLAlchemy."""

    __abstract__ = True

    __table_args__ = {
        "keep_existing": True,
    }

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_desc = Column(String, nullable=True)
    workflow_type = Column(String, nullable=True)
    reward_type = Column(String, nullable=True)


class ExperienceModel(Base):  # type: ignore
    """SQLAlchemy model for Experience."""

    __abstract__ = True

    __table_args__ = {
        "keep_existing": True,
    }

    id = Column(Integer, primary_key=True, autoincrement=True)
    serialized_exp = Column(LargeBinary, nullable=True)
    prompt = Column(String, nullable=True)
    response = Column(String, nullable=True)
    reward = Column(Float, nullable=True)
    consumed = Column(Integer, default=0)
    priority = Column(Float, default=0.0)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.serialized_exp)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            serialized_exp=experience.serialize(),
            reward=experience.reward,
            prompt=experience.prompt_text,
            response=experience.response_text,
        )


class SFTDataModel(Base):  # type: ignore
    """SQLAlchemy model for SFT data."""

    __abstract__ = True

    __table_args__ = {
        "keep_existing": True,
    }

    id = Column(Integer, primary_key=True, autoincrement=True)
    serialized_exp = Column(LargeBinary, nullable=True)
    messages = Column(String, nullable=True)
    consumed = Column(Integer, default=0)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.serialized_exp)

    @classmethod
    def from_messages(
        cls,
        messages: list[dict],
        tokenizer: Any,
        chat_template: Optional[str] = None,
    ) -> "SFTDataModel":
        """Convert a list of messages into a single instance of SFT data."""
        token_ids, action_mask = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=chat_template,
        )
        exp = Experience(
            tokens=token_ids,
            prompt_length=0,
            action_mask=action_mask,
            info={"response_num": sum([1 if m["role"] == "assistant" else 0 for m in messages])},
        )
        return cls(
            serialized_exp=exp.serialize(),
            messages=messages,
        )


class DPODataModel(Base):  # type: ignore
    """SQLAlchemy model for DPO data."""

    __abstract__ = True

    __table_args__ = {
        "keep_existing": True,
    }

    id = Column(Integer, primary_key=True, autoincrement=True)
    serialized_exp = Column(LargeBinary, nullable=True)
    chosen = Column(LargeBinary, nullable=True)
    rejected = Column(LargeBinary, nullable=True)
    consumed = Column(Integer, default=0)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        exp = Experience.deserialize(self.serialized_exp)
        exp.chosen = Experience.deserialize(self.chosen)
        exp.rejected = Experience.deserialize(self.rejected)
        return exp


class SchemaRegistry:
    def __init__(self):
        self.schema_mapping = {
            None: TaskModel,
        }

    def __call__(self, schema, algorithm_type_cls: type = None):
        def _register(cls):
            self.schema_mapping[cls.name()] = schema
            return cls

        if algorithm_type_cls:
            return _register(algorithm_type_cls)
        return _register

    def get_base_class(self, algorithm_type: Union[Algorithm | None]):
        if algorithm_type is not None:
            algorithm_type_name = algorithm_type.name()
        else:
            algorithm_type_name = None
        if algorithm_type_name not in self.schema_mapping:
            raise ValueError(f"Unknown schema: {algorithm_type}")

        return self.schema_mapping[algorithm_type_name]


SCHEMA_REGISTRY = SchemaRegistry()


def create_dynamic_table(algorithm_type: Union[Algorithm | None], table_name: str) -> Any:
    """Create a dynamic table based on the provided algorithm type and table name."""
    base_class = SCHEMA_REGISTRY.get_base_class(algorithm_type)

    table_attrs = {
        "__tablename__": table_name,
    }

    return type(table_name, (base_class,), table_attrs)
