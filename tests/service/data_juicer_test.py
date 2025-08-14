import time
import unittest
from multiprocessing import Process

try:
    import data_juicer  # noqa: [F401]
except ImportError:
    raise ImportError(
        "data_juicer module is not installed. Please install it with `pip install py-data-juicer` to run the tests."
    )
import ray
import torch
from jsonargparse import Namespace

from tests.tools import RayUnittestBaseAysnc, get_template_config
from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.pipelines import ExperiencePipeline
from trinity.common.config import DataJuicerServiceConfig, OperatorConfig
from trinity.common.experience import Experience
from trinity.service.data_juicer.client import DataJuicerClient
from trinity.service.data_juicer.server.server import main
from trinity.service.data_juicer.server.utils import DJConfig, parse_config
from trinity.utils.distributed import get_available_port


class TestDataJuicer(unittest.TestCase):
    def test_config(self):
        trinity_config = {
            "operators": [
                {
                    "llm_quality_score_filter": {
                        "api_or_hf_model": "qwen2.5-7b-instruct",
                        "min_score": 0.0,
                        "input_keys": ["prompt_text"],
                        "field_names": ["prompt", "response"],
                    }
                },
                {
                    "llm_difficulty_score_filter": {
                        "api_or_hf_model": "qwen2.5-7b-instruct",
                        "min_score": 0.0,
                        "enable_vllm": False,
                    }
                },
            ],
            "np": 8,
        }
        config = DJConfig.model_validate(trinity_config)
        dj_config = parse_config(config)
        self.assertIsInstance(dj_config, Namespace)

    def test_server_start(self):
        config = DataJuicerServiceConfig(
            server_url="http://localhost:5005",
            auto_start=False,
        )
        with self.assertRaises(ConnectionError):
            # server is not running, and auto_start is disabled
            # this should raise a ConnectionError
            DataJuicerClient(config)

        # Start the server in a separate process
        def start_server(port):
            server_process = Process(
                target=main, kwargs={"host": "localhost", "port": port, "debug": False}
            )
            server_process.start()
            return server_process

        port = get_available_port()
        config.port = port
        server_process = start_server(port)
        time.sleep(15)  # Wait for the server to start
        config.server_url = f"http://localhost:{port}"
        client = DataJuicerClient(config)
        client.initialize(
            {
                "operators": [
                    {
                        "llm_quality_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "field_names": ["prompt", "response"],
                        }
                    },
                    {
                        "llm_difficulty_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "enable_vllm": False,
                        }
                    },
                ],
                "np": 6,
            }
        )
        self.assertIsNotNone(client.session_id)
        server_process.terminate()
        server_process.join()

        # Test auto start
        config.auto_start = True
        client = DataJuicerClient(config)
        client.initialize(
            {
                "operators": [
                    {
                        "llm_quality_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "input_keys": ["prompt_text"],
                            "field_names": ["prompt", "response"],
                        }
                    },
                    {
                        "llm_difficulty_score_filter": {
                            "api_or_hf_model": "qwen2.5-7b-instruct",
                            "min_score": 0.0,
                            "enable_vllm": False,
                        }
                    },
                ]
            }
        )
        self.assertIsNotNone(client.session_id)
        self.assertIsNotNone(client.server)
        client.close()
        self.assertIsNone(client.session_id)
        self.assertIsNone(client.server)


class TestDataJuicerOperators(RayUnittestBaseAysnc):
    async def test_data_juicer_operators(self):
        config = get_template_config()
        config.service.data_juicer = DataJuicerServiceConfig(
            auto_start=True,
        )
        config.data_processor.experience_pipeline.operators = [
            OperatorConfig(
                name="data_juicer",
                args={
                    "operators": [
                        {
                            "text_length_filter": {
                                "min_len": 10,
                                "max_len": 50,
                                "text_key": "response_text",
                            }
                        },
                        {
                            "word_repetition_filter": {
                                "rep_len": 3,
                                "min_ratio": 0.0,
                                "max_ratio": 0.2,
                                "text_key": "response_text",
                            }
                        },
                    ],
                    "np": 2,
                },
            )
        ]
        config.check_and_update()
        config.buffer.trainer_input.experience_buffer.max_read_timeout = 5
        pipeline = ray.remote(ExperiencePipeline).options(num_cpus=2).remote(config)
        await pipeline.prepare.remote()
        exps = [
            Experience(
                tokens=torch.tensor([1, 2, 3, 4, 5]),
                prompt_length=3,
                prompt_text="Hello, how are you?",
                response_text="Hi, I am fine.",
            ),
            Experience(  # too short response
                tokens=torch.tensor([1, 2, 3, 4, 5]),
                prompt_length=3,
                prompt_text="What is your name?",
                response_text="Trinity.",
            ),
            Experience(  # repeated words
                tokens=torch.tensor([1, 2, 3, 4, 5]),
                prompt_length=3,
                prompt_text="What day is it today?",
                response_text="Today is Sunday Sunday Sunday Sunday Sunday and it's a happy day!",
            ),
            Experience(
                tokens=torch.tensor([1, 2, 3, 4, 5]),
                prompt_length=3,
                prompt_text="What is your favorite color?",
                response_text="My favorite color is blue.",
            ),
        ]
        metrics = await pipeline.process.remote(exps)
        self.assertIsInstance(metrics, dict)
        reader = get_buffer_reader(config.buffer.trainer_input.experience_buffer, config.buffer)
        filtered_exps = reader.read(batch_size=2)
        self.assertEqual(len(filtered_exps), 2)
        with self.assertRaises(TimeoutError):
            reader.read(batch_size=1)
        await pipeline.close.remote()
