import io
import json
import time
from multiprocessing import Process, set_start_method
from typing import Dict, List, Tuple

import pyarrow as pa
import requests
from datasets import Dataset

from trinity.common.config import DataJuicerServiceConfig
from trinity.common.experience import Experience, from_hf_datasets, to_hf_datasets
from trinity.utils.distributed import get_available_port, is_port_available
from trinity.utils.log import get_logger


class DataJuicerClient:
    """Client for interacting with the DataJuicer server."""

    def __init__(self, config: DataJuicerServiceConfig):
        self.logger = get_logger(__name__, in_ray_actor=True)
        self.config = config
        self.url = config.server_url
        self.session_id = None
        self.server = None
        if not self.config.auto_start:
            # If auto-start is disabled, check the connection immediately
            self._check_connection()

    def _start_server(self):
        """Start the DataJuicer server."""
        if not self.config.auto_start:
            # Server auto-start is disabled, use the provided URL
            return None

        from trinity.service.data_juicer.server.server import main

        if not self.config.port:
            self.config.port = get_available_port()
        elif not is_port_available(self.config.port):
            self.config.port = get_available_port()
        self.logger.info(
            f"Starting DataJuicer server at {self.config.server_url} on port {self.config.port}"
        )
        self.url = f"http://localhost:{self.config.port}"
        set_start_method("spawn", force=True)
        server_process = Process(
            target=main, kwargs={"host": "localhost", "port": self.config.port, "debug": False}
        )
        server_process.start()
        # Wait for the server to start
        while True:
            try:
                if self._check_connection():
                    break
            except ConnectionError:
                time.sleep(5)
        self.logger.info(f"DataJuicer server at {self.url} started successfully.")
        return server_process

    def _check_connection(self) -> bool:
        """Check if the DataJuicer server is reachable."""
        try:
            response = requests.get(f"{self.url}/health")  # Check if the server is running
        except Exception as e:
            raise ConnectionError(f"Failed to connect to DataJuicer server at {self.url}: {e}")
        if response.status_code != 200:
            raise ConnectionError(
                f"DataJuicer server at {self.url} is not reachable. Status code: {response.status_code}"
            )
        return True

    def initialize(self, config: dict):
        self.server = self._start_server()
        response = requests.post(f"{self.url}/create", json=config)
        response.raise_for_status()
        self.session_id = response.json().get("session_id")

    def process_experience(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        st = time.time()
        if not self.session_id:
            raise ValueError("DataJuicer session is not initialized.")

        dataset = to_hf_datasets(exps)

        arrow_bytes = serialize_dataset_to_arrow(dataset)
        files = {"arrow_data": ("dataset.arrow", arrow_bytes, "application/octet-stream")}
        data = {"session_id": self.session_id}
        response = requests.post(f"{self.url}/process_experience", data=data, files=files)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to process experiences: {response.status_code}, {response.json().get('error')}"
            )
        metrics = json.loads(response.headers.get("X-Metrics"))
        dataset = deserialize_arrow_to_dataset(response.content)
        exps = from_hf_datasets(dataset)
        # move all computed stats into the info field of experiences
        for exp, sample in zip(exps, dataset):
            if "__dj__stats__" not in sample:
                continue
            if exp.info is None:
                exp.info = {}
            for stats_key in sample["__dj__stats__"]:
                exp.info[stats_key] = sample["__dj__stats__"][stats_key]
        metrics["time/dj_process_experience"] = time.time() - st
        return exps, metrics

    def process_task(self) -> Dict:
        """Process a task using the Data-Juicer service."""
        if not self.session_id:
            raise ValueError("DataJuicer session is not initialized.")
        json_data = {"session_id": self.session_id}
        response = requests.post(f"{self.url}/process_task", json=json_data)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to process task: {response.status_code}, {response.json().get('error')}"
            )
        return response.json().get("metrics")

    def close(self):
        """Close the DataJuicer client connection."""
        if self.session_id:
            response = requests.post(f"{self.url}/close", json={"session_id": self.session_id})
            response.raise_for_status()
            self.session_id = None
        if self.server:
            self.server.terminate()
            self.server.join()
            self.server = None


def serialize_dataset_to_arrow(dataset):
    arrow_table = dataset.data.table

    buffer = io.BytesIO()
    with pa.ipc.new_stream(buffer, arrow_table.schema) as writer:
        writer.write_table(arrow_table)

    arrow_bytes = buffer.getvalue()

    return arrow_bytes


def deserialize_arrow_to_dataset(arrow_bytes):
    buffer = io.BytesIO(arrow_bytes)

    with pa.ipc.open_stream(buffer) as reader:
        arrow_table = reader.read_all()

    dataset = Dataset(arrow_table)

    return dataset
