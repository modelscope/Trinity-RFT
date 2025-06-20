"""Writer of the SQL buffer."""

import ray

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.ray_wrapper import DBWrapper
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType


class SQLWriter(BufferWriter):
    """Writer of the SQL buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig) -> None:
        assert meta.storage_type == StorageType.SQL
        # we only support write RFT algorithm buffer for now
        self.wrap_in_ray = meta.wrap_in_ray
        self.db_wrapper = DBWrapper.get_wrapper(meta, config)

    def write(self, data: list) -> None:
        if self.wrap_in_ray:
            ray.get(self.db_wrapper.write.remote(data))
        else:
            self.db_wrapper.write(data)

    def finish(self) -> None:
        # TODO: implement this
        pass
