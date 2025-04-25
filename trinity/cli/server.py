import traceback

import fire
from flask import Flask, jsonify, request

app = Flask(__name__)

APP_NAME = "trinity_rft"


@app.route(f"/{APP_NAME}", methods=["GET"])
def trinity_training():
    config_path = request.args.get("configPath")
    try:
        from trinity.cli.launcher import run

        run(config_path)
        ret = 0
        msg = "Training Success."
    except:  # noqa: E722
        traceback.print_exc()
        msg = traceback.format_exc()
        ret = 1
    return jsonify({"return_code": ret, "message": msg})


def main(port=5006):
    app.run(port=port, debug=True)


if __name__ == "__main__":
    fire.Fire(main)
