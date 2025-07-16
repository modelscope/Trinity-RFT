import threading
from typing import List

import fire
import ray
from flask import Flask, jsonify, request
from markupsafe import escape

app = Flask(__name__)

APP_NAME = "data_processor"

# event pool for stopping all running services
EVENT_POOL: List[threading.Event] = []

@app.route(f"/{APP_NAME}/<pipeline_type>", methods=["GET"])
def data_processor(pipeline_type):
    from trinity.common.config import load_config
    from trinity.data.controllers.active_iterator import DataActiveIterator
    from trinity.data.utils import safe_str_to_bool, get_explorer_model_service

    # the path to the config file
    config_path = request.args.get("configPath")
    # whether the explorer is synced. Only need for dynamic task pipeline.
    is_sync = safe_str_to_bool(request.args.get("is_sync"), default=True)

    pipeline_type = escape(pipeline_type)
    config = load_config(config_path)
    config.check_and_update()

    # init ray
    ray.init(namespace=config.ray_namespace, ignore_reinit_error=True)

    pipeline_config = getattr(config.data_processor, pipeline_type)
    if pipeline_config is None:
        return jsonify(
            {
                "return_code": -1,
                "message": f"Error: {pipeline_type} is not supported or the corresponding config is empty",
            }
        )

    if pipeline_config.dj_config_path is None and pipeline_config.dj_process_desc is None:
        return jsonify(
            {
                "return_code": -1,
                "message": "Error: Both dj_config_path and dj_process_desc in the pipeline config are None.",
            }
        )

    if pipeline_type == "task_pipeline":
        # must be sync
        # if it's a dynamic task pipeline, try to get the api url and model path from the explorer
        api_url, model_path = get_explorer_model_service(config)
        updated_api_info = {
            "base_url": api_url,
            "api_key": "EMPTY",
            "model": model_path,
        }
        iterator = DataActiveIterator(pipeline_config, config.buffer, pipeline_type=pipeline_type, updated_api_info=updated_api_info)
        # If the explorer is synced, the scorer model is updated, so the priority and the corresponding stats/meta need
        # to be recomputed.
        ret, msg = iterator.run(recompute=is_sync)
        return jsonify({"return_code": ret, "message": msg})
    elif pipeline_type == "experience_pipeline":
        # must be async
        iterator = DataActiveIterator(pipeline_config, config.buffer, pipeline_type=pipeline_type)
        # add an event
        event = threading.Event()
        thread = threading.Thread(target=iterator.run, args=(event, False,))
        thread.start()
        # add this event to the event pool
        EVENT_POOL.append(event)
        return jsonify({"return_code": 0, "message": "Experience pipeline starts successfully."})


@app.route(f"/{APP_NAME}/stop_all", methods=["GET"])
def stop_all():
    try:
        # stop all services
        for event in EVENT_POOL:
            event.set()

        # clear all temp file buffers
        from trinity.data.utils import clear_temp_file_buffers
        clear_temp_file_buffers()
    except Exception:
        import traceback

        traceback.print_exc()
        return jsonify({"return_code": 1, "message": traceback.format_exc()})
    return jsonify({"return_code": 0, "message": "All data pipelines are stopped."})


def main(port=5005):
    app.run(port=port, debug=True)


if __name__ == "__main__":
    fire.Fire(main)
