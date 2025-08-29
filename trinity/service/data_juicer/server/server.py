import argparse
import json
import traceback
import uuid

from flask import Flask, jsonify, make_response, request

from trinity.service.data_juicer.client import (
    deserialize_arrow_to_dataset,
    serialize_dataset_to_arrow,
)
from trinity.service.data_juicer.server.session import DataJuicerSession
from trinity.service.data_juicer.server.utils import DJConfig

app = Flask(__name__)
sessions = {}


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is running."""
    return jsonify({"status": "ok"}), 200


@app.route("/create", methods=["POST"])
def create():
    """Create a new data juicer session.

    Args:
        config (dict): Configuration parameters for the session.

    For example, the config should look like this:

    .. code-block:: python

        {
            "operators": [
                {
                    "operator1_name": {
                        "arg1": "value1",
                        "arg2": "value2"
                    }
                },
                {
                    "operator2_name": {
                        "arg1": "value1",
                        "arg2": "value2"
                    }
                }
            ],
            "config_path": "path/to/data_juicer_config.yaml",
        }
    """
    config = request.json
    try:
        config = DJConfig.model_validate(config)
    except Exception as e:
        return jsonify({"error": f"Failed to parse config: {e}"}), 400

    session_id = str(uuid.uuid4())
    sessions[session_id] = DataJuicerSession(config)

    return jsonify({"session_id": session_id, "message": "Session created successfully."})


@app.route("/process_experience", methods=["POST"])
def process_experience():
    """
    Process uploaded experiences for a given session.
    Expects a multipart/form-data POST with arrow bytes and session_id.
    Returns processed experiences and metrics as arrow bytes and JSON.
    """
    session_id = request.form.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session ID not found."}), 404

    # Check for file in request
    if "arrow_data" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    arrow_file = request.files["arrow_data"]
    arrow_bytes = arrow_file.read()
    ds = deserialize_arrow_to_dataset(arrow_bytes)

    # Process experiences using session
    session = sessions[session_id]
    try:
        # from_hf_datasets and to_hf_datasets should be imported from trinity.common.experience
        processed_ds, metrics = session.process_experience(ds)
    except Exception:
        return jsonify({"error": f"Experience processing failed:\n{traceback.format_exc()}"}), 500

    # Serialize processed experiences to parquet in-memory
    return_bytes = serialize_dataset_to_arrow(processed_ds)

    # Return arrow bytes and metrics as response
    response = make_response(return_bytes)
    response.headers["X-Metrics"] = json.dumps(metrics)
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Content-Disposition"] = "attachment; filename=processed.arrow"
    return response


@app.route("/process_task", methods=["POST"])
def process_task():
    """Process a task for a given session.
    Different from process_experience which process a batch of experiences,
    this endpoint process a whole taskset.
    """
    data = request.json
    session_id = data.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session ID not found."}), 404

    session = sessions[session_id]
    try:
        metrics = session.process_task()
        return jsonify({"metrics": metrics})
    except Exception:
        return jsonify({"error": f"Task processing failed:\n{traceback.format_exc()}"}), 500


@app.route("/close", methods=["POST"])
def close():
    """Close a data juicer session."""
    data = request.json
    session_id = data.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session ID not found."}), 404

    del sessions[session_id]
    return jsonify({"message": "Session closed successfully."})


def main(host="localhost", port=5005, debug=False):
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=5005, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(host=args.host, port=args.port, debug=args.debug)
