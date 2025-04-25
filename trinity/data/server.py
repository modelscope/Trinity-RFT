import fire
from flask import Flask, jsonify, request

app = Flask(__name__)

APP_NAME = "data_workflow"


@app.route(f"/{APP_NAME}", methods=["GET"])
def data_workflow():
    from trinity.common.config import load_config
    from trinity.data.controllers.active_iterator import DataActiveIterator

    config_path = request.args.get("configPath")
    config = load_config(config_path)

    iterator = DataActiveIterator(config)
    ret, msg = iterator.run()
    return jsonify({"return_code": ret, "message": msg})


def main(port=5005):
    app.run(port=port, debug=True)


if __name__ == "__main__":
    fire.Fire(main)
