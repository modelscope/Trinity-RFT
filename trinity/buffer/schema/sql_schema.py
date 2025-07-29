"""Schema for SQLAlchemy models."""

from typing import Any, Optional, Union

from sqlalchemy import Column, Float, Integer, LargeBinary, String
from sqlalchemy.ext.declarative import declarative_base

from trinity.common.experience import Experience

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
        from trinity.common.models.utils import tokenize_and_mask_messages_hf

        tokens, action_mask, prompt_length = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=chat_template,
        )
        exp = Experience(
            tokens=tokens,
            action_mask=action_mask,
            prompt_length=prompt_length,
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


def create_dynamic_table(algorithm_type: Union[str | None], table_name: str) -> Any:
    """Create a dynamic table based on the provided algorithm type and table name."""
    if algorithm_type is None:
        base_class = TaskModel
    else:
        from trinity.algorithm.algorithm import ALGORITHM_TYPE

        algorithm = ALGORITHM_TYPE.get(algorithm_type)
        base_class = algorithm.schema

    table_attrs = {
        "__tablename__": table_name,
    }

    return type(table_name, (base_class,), table_attrs)
