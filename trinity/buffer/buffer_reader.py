"""Reader of the buffer."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BufferReader(ABC):
    """Interface of the buffer reader."""

    @abstractmethod
    def read(self, batch_size: Optional[int] = None) -> List:
        """Read from buffer."""

    @abstractmethod
    async def read_async(self, batch_size: Optional[int] = None) -> List:
        """Read from buffer asynchronously."""

    def __len__(self) -> int:
        """Get the number of samples in buffer."""
        raise NotImplementedError

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state_dict: Dict) -> None:
        pass
