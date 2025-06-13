import fire
from flask import Flask, jsonify, request
from markupsafe import escape

app = Flask(__name__)

APP_NAME = "data_workflow"


@app.route(f"/{APP_NAME}/<pipeline_type>", methods=["GET"])
def data_workflow(pipeline_type):
    from trinity.common.config import load_config
    from trinity.data.controllers.active_iterator import DataActiveIterator

    config_path = request.args.get("configPath")
    pipeline_type = escape(pipeline_type)
    config = load_config(config_path)

    pipeline_config = getattr(config, pipeline_type)
    if pipeline_config is None:
        return jsonify(
            {
                "return_code": -1,
                "message": f"{pipeline_type} is not supported or the corresponding config is empty",
            }
        )

    iterator = DataActiveIterator(pipeline_config, config.buffer)
    ret, msg = iterator.run()
    return jsonify({"return_code": ret, "message": msg})


def main(port=5005):
    app.run(port=port, debug=True)


if __name__ == "__main__":
    fire.Fire(main)
