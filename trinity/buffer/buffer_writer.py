"""Writer of the buffer."""
from abc import ABC, abstractmethod
from typing import List


class BufferWriter(ABC):
    """Interface of the buffer writer."""

    @abstractmethod
    def write(self, data: List) -> None:
        """Write to buffer."""

    @abstractmethod
    def finish(self) -> None:
        """Finish writing."""
