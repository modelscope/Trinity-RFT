from typing import List

from sqlalchemy import asc, create_engine, desc
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from trinity.buffer.utils import retry_session
from trinity.common.config import DataProcessorConfig
from trinity.common.schema import Base, RftDatasetModel
from trinity.data.core.dataset import RftDataset
from trinity.utils.log import get_logger

logger = get_logger(__name__)


def rft_dataset_to_model(dataset: RftDataset) -> List[RftDatasetModel]:
    # hit keys of schema
    hit_schema_keys = []
    hit_dataset_keys = []
    # get hit keys & vals
    # - for content keys, we need to map it with content_key_mapping and try to
    #   find them in the dataset
    # - for other keys, we just need to check if they are in the dataset
    data = dataset.data
    features = data.features
    content_key_mapping = dataset.format.__dict__
    schema_keys = {key for key in RftDatasetModel.__dict__.keys() if not key.startswith("_")}
    for schema_key in schema_keys:
        key = schema_key
        if f"{schema_key}_key" in content_key_mapping:
            key = content_key_mapping[f"{schema_key}_key"]
        if key in features:
            hit_schema_keys.append(schema_key)
            hit_dataset_keys.append(key)
    # construct entries
    entries = []
    for sample in data:
        valid_data = {
            schema_key: sample[key] for schema_key, key in zip(hit_schema_keys, hit_dataset_keys)
        }
        entries.append(RftDatasetModel(**valid_data))
    return entries


class RftDatasetDB:
    def __init__(self, config: DataProcessorConfig) -> None:
        self.db_url = config.db_url
        self.engine = create_engine(self.db_url, poolclass=NullPool)
        self.config = config
        try:
            Base.metadata.create_all(self.engine, checkfirst=True)
        except OperationalError:
            logger.warning("Failed to create database, assuming it already exists.")
        self.session = sessionmaker(bind=self.engine)

    def add_entries(self, dataset: RftDataset):
        with retry_session(
            self, self.config.max_retry_times, self.config.max_retry_interval
        ) as session:
            session.add_all(rft_dataset_to_model(dataset))

    def get_entries(self, num_entries: int, order_by: str = None, ascending: bool = False):
        # get num_entries entries from the database
        if order_by is not None and hasattr(RftDatasetModel, order_by):
            order_by_key = getattr(RftDatasetModel, order_by)
            order_by_key = asc(order_by_key) if ascending else desc(order_by_key)
        else:
            order_by_key = None
        with retry_session(
            self, self.config.max_retry_times, self.config.max_retry_interval
        ) as session:
            entries = (
                session.query(RftDatasetModel)
                .order_by(order_by_key)
                .limit(num_entries)
                .with_for_update()
                .all()
            )

            for entry in entries:
                entry.consumed_cnt += 1
            samples = [entry.to_dict() for entry in entries]
            return samples
