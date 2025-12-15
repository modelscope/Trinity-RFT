"""Reader of the SQL buffer."""

from typing import Dict, List, Optional

import ray

from trinity.buffer.buffer_reader import BufferReader
from trinity.buffer.storage.sql import SQLStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class SQLReader(BufferReader):
    """Reader of the SQL buffer."""

    def __init__(self, config: StorageConfig) -> None:
        assert config.storage_type == StorageType.SQL.value
        self.wrap_in_ray = config.wrap_in_ray
        self.storage = SQLStorage.get_wrapper(config)

    def read(self, batch_size: Optional[int] = None) -> List:
        if self.wrap_in_ray:
            return ray.get(self.storage.read.remote(batch_size))
        else:
            return self.storage.read(batch_size)

    async def read_async(self, batch_size: Optional[int] = None) -> List:
        if self.wrap_in_ray:
            try:
                return ray.get(self.storage.read.remote(batch_size))
            except StopIteration:
                raise StopAsyncIteration
        else:
            return self.storage.read(batch_size)

    def state_dict(self) -> Dict:
        # SQL Not supporting state dict yet
        return {"current_index": 0}

    def load_state_dict(self, state_dict):
        # SQL Not supporting state dict yet
        return None
