from typing import Any, Dict

from trinity.common.config import Config, OperatorConfig, TaskPipelineConfig
from trinity.utils.log import get_logger


def check_and_run_task_pipeline(config: Config) -> Dict:
    if config.mode not in {"explore", "train", "both"}:
        return {}
    if config.data_processor.task_pipeline is None:
        return {}

    task_pipeline = TaskPipeline(config)
    try:
        return task_pipeline.process()
    except Exception as e:
        raise RuntimeError(f"Task pipeline failed: {e}")
    finally:
        task_pipeline.close()


class TaskPipeline:
    """
    A class to process task datasets through DataJuicer.
    """

    def __init__(self, config: Config):
        self.logger = get_logger(__name__)
        from trinity.service.data_juicer.client import DataJuicerClient

        self.client = DataJuicerClient(config.service.data_juicer)  # type: ignore [arg-type]
        self.pipeline_config = config.data_processor.task_pipeline

    def convert_pipeline_config(self, pipeline_config: TaskPipelineConfig) -> Dict[str, Any]:
        """
        Convert the TaskPipelineConfig to a format suitable for DataJuicer.
        """

        def _convert_operator(operator: OperatorConfig) -> Dict:
            return {operator.name: {key: value for key, value in operator.args.items()}}

        if pipeline_config.output.path is None:
            raise ValueError("When using task pipeline, taskset.path must be set.")

        converted_config = {
            "pipeline_type": "task",
            "operators": [_convert_operator(op) for op in pipeline_config.operators],
            "np": pipeline_config.num_process,
            "config_path": pipeline_config.config_path,
            "inputs": [path for path in pipeline_config.inputs],
            "target_fields": pipeline_config.target_fields,
            "output_dir": pipeline_config.output.path,
            "priority_weights": pipeline_config.priority_weights,
            "top_k": pipeline_config.top_k,
        }
        return converted_config

    def process(self) -> Dict[str, Any]:
        """
        Process the task datasets using DataJuicer.

        Returns:
            Dict[str, Any]: Metrics for logging.
        """
        # Convert the pipeline configuration
        converted_config = self.convert_pipeline_config(self.pipeline_config)  # type: ignore [arg-type]
        self.client.initialize(converted_config)
        self.logger.info("Starting task processing...")
        metrics = self.client.process_task()
        self.logger.info("Task processing completed.")
        return metrics

    def close(self):
        self.client.close()
