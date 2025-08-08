from trinity.common.config import DataPipelineConfig, DataProcessorConfig
from trinity.common.constants import DataProcessorPipelineType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


def check_and_activate_data_processor(data_processor_config: DataProcessorConfig, config_path: str):
    if (
        data_processor_config.data_processor_url is not None
        and data_processor_config.task_pipeline is not None
        and validate_data_pipeline(
            data_processor_config.task_pipeline, DataProcessorPipelineType.TASK
        )
    ):
        activate_data_processor(
            f"{data_processor_config.data_processor_url}/{DataProcessorPipelineType.TASK.value}",
            config_path,
        )
    # TODO: check and activate experience pipeline


def activate_data_processor(data_processor_url: str, config_path: str):
    """Check whether to activate data module and preprocess datasets."""
    from trinity.cli.client import request

    logger.info(f"Activating data module of {data_processor_url}...")
    res = request(
        url=data_processor_url,
        configPath=config_path,
    )
    if res["return_code"] != 0:
        logger.error(f"Failed to activate data module: {res['return_msg']}.")
        return


def stop_data_processor(base_data_processor_url: str):
    """Stop all pipelines in the data processor"""
    from trinity.cli.client import request

    logger.info(f"Stopping all pipelines in {base_data_processor_url}...")
    res = request(url=f"{base_data_processor_url}/stop_all")
    if res["return_code"] != 0:
        logger.error(f"Failed to stop all data pipelines: {res['return_msg']}.")
        return


def validate_data_pipeline(
    data_pipeline_config: DataPipelineConfig, pipeline_type: DataProcessorPipelineType
):
    """
    Check if the data pipeline is valid. The config should:
    1. Non-empty input buffer
    2. Different input/output buffers

    :param data_pipeline_config: the input data pipeline to be validated.
    :param pipeline_type: the type of pipeline, should be one of DataProcessorPipelineType
    """
    input_buffers = data_pipeline_config.input_buffers
    output_buffer = data_pipeline_config.output_buffer
    # common checks
    # check if the input buffer list is empty
    if len(input_buffers) == 0:
        logger.warning("Empty input buffers in the data pipeline. Won't activate it.")
        return False
    # check if the input and output buffers are different
    input_buffer_names = [buffer.name for buffer in input_buffers]
    if output_buffer.name in input_buffer_names:
        logger.warning("Output buffer exists in input buffers. Won't activate it.")
        return False
    if pipeline_type == DataProcessorPipelineType.TASK:
        # task pipeline specific
        # "raw" field should be True for task pipeline because the data source must be raw data files
        for buffer in input_buffers:
            if not buffer.raw:
                logger.warning(
                    'Input buffers should be raw data files for task pipeline ("raw" field should be True). Won\'t activate it.'
                )
                return False
    elif pipeline_type == DataProcessorPipelineType.EXPERIENCE:
        # experience pipeline specific
        # No special items need to be checked.
        pass
    else:
        logger.warning(f"Invalid pipeline type: {pipeline_type}..")
        return False
    return True
