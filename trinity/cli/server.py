import traceback
from flask import Flask, jsonify, request

app = Flask(__name__)

APP_NAME = 'trinity_training'
PORT = 5006

@app.route(f"/{APP_NAME}", methods=["GET"])
def trinity_training():
    config_path = request.args.get("configPath")
    try:
        from trinity.cli.launcher import run

        run(config_path)
        ret = 0
        msg = "Training Success."
    except:
        traceback.print_exc()
        msg = traceback.format_exc()
        ret = 1
    return jsonify({"return_code": ret, "message": msg})


if __name__ == "__main__":
    app.run(port=PORT, debug=True)
