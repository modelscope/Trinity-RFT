"""Tests for explorer."""
import asyncio
import json
import multiprocessing
import os
import random
import shutil
from datetime import datetime

import httpx
import openai
import ray

from tests.tools import (
    RayUnittestBase,
    RayUnittestBaseAysnc,
    TensorBoardParser,
    get_api_model_path,
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.buffer import get_buffer_reader
from trinity.cli.launcher import explore, run_stage
from trinity.common.config import ExperienceBufferConfig, InferenceModelConfig
from trinity.common.constants import StorageType
from trinity.explorer.explorer import Explorer
from trinity.manager.state_manager import StateManager


class BaseExplorerCase(RayUnittestBase):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "explore"
        self.config.buffer.total_epochs = 2
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.algorithm.repeat_times = 2
        self.config.monitor.monitor_type = "tensorboard"
        self.config.project = "Trinity-unittest"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 2
        self.config.explorer.eval_interval = 4


class TestExplorerCountdownEval(BaseExplorerCase):
    def test_explorer(self):
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.eval_tasksets.extend(
            [
                get_unittest_dataset_config("countdown", "test"),
                get_unittest_dataset_config("eval_short"),
                get_unittest_dataset_config("eval_long"),
            ]
        )
        self.config.name = f"explore-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)
        self.assertTrue("eval/eval_short/accuracy/max" in eval_metrics)
        self.assertTrue("eval/eval_long/accuracy/max" in eval_metrics)


class TestExplorerGSM8KRULERNoEval(BaseExplorerCase):
    def test_explorer(self):
        self.config.explorer.rollout_model.engine_num = 2
        self.config.explorer.auxiliary_models = [
            InferenceModelConfig(
                model_path=get_api_model_path(),
                tensor_parallel_size=1,
                engine_num=2,
            )
        ]
        self.config.algorithm.repeat_times = 2
        self.config.buffer.total_steps = 2
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k_ruler")
        self.config.name = f"explore-no-eval-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "std_threshold": 0.0001,
        }
        self.config.check_and_update()
        explore(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) == 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 2)


class TestExplorerGSM8k(BaseExplorerCase):
    def test_explorer(self):
        self.config.algorithm.repeat_times = 2
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.name = f"explore-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # some step may be skipped due to same reward
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "epsilon": 1e-6,
        }
        self.config.model.max_model_len = 10240
        self.config.model.max_response_tokens = 8192
        self.config.model.min_response_tokens = 8192
        self.config.explorer.rollout_model.ignore_eos = True
        self.config.check_and_update()
        explorer = Explorer.get_actor(self.config)
        ray.get(explorer.prepare.remote())
        ray.get(explorer.sync_weight.remote())
        ray.get(explorer.explore.remote())
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) == 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        self.assertTrue(parser.metric_exist("experience_pipeline/experience_count"))
        experience_counts = parser.metric_values("experience_pipeline/experience_count")
        self.assertTrue(len(experience_counts) == 4)
        for count in experience_counts:
            self.assertTrue(count >= 0)
            self.assertTrue(count <= 2 * 4)  # repeat_times * batch_size
            self.assertTrue(count % 2 == 0)  # should be multiple of repeat_times
        exp_save_path = self.config.buffer.trainer_input.experience_buffer.path
        with open(exp_save_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertTrue(len(lines) <= 4 * 2 * 4)  # step * repeat_times * batch_size
            self.assertTrue(len(lines) % (2 * 4) == 0)
            exp = json.loads(lines[0])
            self.assertEqual(exp["response_length"], 8192)
        ray.get(explorer.shutdown.remote())


def run_serve(config):
    config.check_and_update()
    run_stage(config)


def run_agent(base_url, model_path: str):
    client = openai.Client(base_url=base_url, api_key="testkey")
    contents = [
        "Hello, how are you?",
        "What is the capital of China?",
        "Tell me a joke.",
        "Explain the theory of relativity.",
        "What is the meaning of life?",
        "How does a computer work?",
        "What is the weather like today?",
        "Can you recommend a good book?",
        "What is the best way to learn programming?",
        "Describe the process of photosynthesis.",
    ]
    response = client.chat.completions.create(
        model=model_path,
        messages=[{"role": "user", "content": random.choice(contents)}],
    )
    return response.choices[0].message.content


class ServeTest(RayUnittestBaseAysnc):
    def setUp(self):
        self.config = get_template_config()
        self.config.mode = "serve"
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm"
        self.config.algorithm.repeat_times = 1
        self.config.monitor.monitor_type = "tensorboard"
        self.config.project = "Trinity-unittest"
        self.config.explorer.rollout_model.engine_num = 4
        self.config.explorer.rollout_model.enable_openai_api = True
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.explorer.api_port = 8010
        self.config.explorer.service_status_check_interval = 30
        self.config.buffer.trainer_input.experience_buffer = ExperienceBufferConfig(
            name="experience_buffer",
            storage_type=StorageType.SQL,
        )
        self.config.check_and_update()
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

    async def test_serve(self):  # noqa: C901
        serve_process = multiprocessing.Process(target=run_serve, args=(self.config,))
        serve_process.start()
        await asyncio.sleep(10)

        state_manager = StateManager(
            path=self.config.checkpoint_job_dir,
            explorer_name=self.config.explorer.name,
        )

        # wait for explorer initialization
        for i in range(30):
            try:
                server_url = state_manager.load_explorer_server_url()
            except Exception:
                server_url = None
            if server_url:
                break
            await asyncio.sleep(3)
        if not server_url:
            raise RuntimeError("Explorer server URL not found.")
        # wait for server setup
        for i in range(10):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{server_url}/health")
                    if response.status_code == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(2)

        task_num = 10
        apps = []
        for i in range(task_num):
            app_process = multiprocessing.Process(
                target=run_agent, args=(server_url + "/v1", self.config.model.model_path)
            )
            apps.append(app_process)
            app_process.start()

        for app in apps:
            app.join(timeout=60)
            self.assertFalse(app.is_alive())

        finish_step = None

        for i in range(20):
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{server_url}/metrics")
                self.assertEqual(response.status_code, 200)
                metrics = response.json()
                metrics_keys = list(metrics.keys())
                self.assertIn("explore_step_num", metrics_keys)
                self.assertIn("rollout/total_experience_count", metrics_keys)
                self.assertIn("rollout/model_0/total_request_count", metrics_keys)
                self.assertIn("rollout/model_3/model_version", metrics_keys)
                if not finish_step and metrics["rollout/total_experience_count"] == task_num:
                    finish_step = metrics["explore_step_num"]
                if finish_step and metrics["explore_step_num"] >= finish_step + 1:
                    # wait for one more step to ensure all data are written to buffer
                    break
            await asyncio.sleep(3)

        serve_process.terminate()
        serve_process.join(timeout=10)

        # check buffer
        self.config.buffer.trainer_input.experience_buffer.max_read_timeout = 5
        buffer_reader = get_buffer_reader(
            self.config.buffer.trainer_input.experience_buffer,
        )
        exps = await buffer_reader.read_async(batch_size=10)
        for exp in exps:
            self.assertTrue(len(exp.tokens) > 0)
        self.assertEqual(len(exps), task_num)

    def tearDown(self):
        shutil.rmtree(self.config.checkpoint_job_dir, ignore_errors=True)
