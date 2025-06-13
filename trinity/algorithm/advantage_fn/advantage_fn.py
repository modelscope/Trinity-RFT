from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from trinity.utils.registry import Registry

ADVANTAGE_FN = Registry("advantage_fn")


class AdvantageFn(ABC):
    @abstractmethod
    def __call__(self, exps: Any, **kwargs: Dict) -> Tuple[Any, Dict]:
        """Calculate advantages from experiences

        Args:
            exps (`DataProto`): The input experiences.
            kwargs (`Dict`): The step-level parameters for calculating advantages.

        Returns:
            `DataProto`: The experiences with advantages.
            `Dict`: The metrics for logging.
        """

    @classmethod
    @abstractmethod
    def default_args(cls) -> Dict:
        """
        Returns:
            `Dict`: The default init arguments for the advantage function.
        """
