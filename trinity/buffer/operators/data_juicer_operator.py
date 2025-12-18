from typing import Dict, List, Optional, Tuple

from trinity.buffer.operators.experience_operator import ExperienceOperator
from trinity.common.config import DataJuicerServiceConfig
from trinity.common.experience import Experience
from trinity.service.data_juicer.client import DataJuicerClient


class DataJuicerOperator(ExperienceOperator):
    def __init__(
        self,
        service_config: DataJuicerServiceConfig,
        operators: Optional[List[Dict]] = None,
        config_path: Optional[str] = None,
        np: int = 4,
    ):
        """
        Initialize the DataJuicerOperator.

        Args:
            service_config (config): The configuration for the DataJuicer service.
            operators(`List[Dict]`): A list of operators with their configurations.
            config_path(`str`): Path to the Data-Juicer configuration file.
            np (`int`): Number of processes to use for Data-Juicer. Default is 4.

        Note:
            - Must include one of the following, and the priority is from high to low:
                - `config_path` (`str`)
                - `operators` (`List[Dict]`)
        """
        self.client = DataJuicerClient(config=service_config)
        self.client.initialize(
            {
                "operators": operators,
                "config_path": config_path,
                "np": np,
                "pipeline_type": "experience",
            }
        )

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        return self.client.process_experience(exps)

    def close(self):
        """Close the DataJuicer client connection."""
        self.client.close()
