"""Reader of the buffer."""
from abc import ABC, abstractmethod
from typing import List, Optional

from trinity.common.constants import ReadStrategy


class BufferReader(ABC):
    """Interface of the buffer reader."""

    @abstractmethod
    def read(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        """Read from buffer."""

    @abstractmethod
    async def read_async(
        self, batch_size: Optional[int] = None, strategy: Optional[ReadStrategy] = None
    ) -> List:
        """Read from buffer asynchronously."""
