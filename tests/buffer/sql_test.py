import os

import ray
import torch
from parameterized import parameterized

from tests.tools import RayUnittestBaseAsync
from trinity.buffer import get_buffer_reader
from trinity.buffer.reader.sql_reader import SQLReader
from trinity.buffer.writer.sql_writer import SQLWriter
from trinity.common.config import (
    ExperienceBufferConfig,
    ReplayBufferConfig,
    TasksetConfig,
)
from trinity.common.constants import StorageType
from trinity.common.experience import Experience

db_path = os.path.join(os.path.dirname(__file__), "test.db")


class TestSQLBuffer(RayUnittestBaseAsync):
    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    async def test_sql_exp_buffer_read_write(self, enable_replay: bool) -> None:
        total_num = 8
        put_batch_size = 2
        read_batch_size = 4
        config = ExperienceBufferConfig(
            name="test_buffer",
            schema_type="experience",
            path=f"sqlite:///{db_path}",
            storage_type=StorageType.SQL.value,
            batch_size=read_batch_size,
            max_read_timeout=3,
        )
        if enable_replay:
            config.replay_buffer = ReplayBufferConfig(enable=True)
        sql_writer = SQLWriter(config.to_storage_config())
        sql_reader = SQLReader(config.to_storage_config())
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                prompt_length=i,
                reward=float(i),
                logprobs=torch.tensor([0.1]),
                info={"model_version": i},
            )
            for i in range(1, put_batch_size + 1)
        ]
        self.assertEqual(await sql_writer.acquire(), 1)
        for _ in range(total_num // put_batch_size):
            await sql_writer.write_async(exps)
        for _ in range(total_num // read_batch_size):
            exps = sql_reader.read()
            self.assertEqual(len(exps), read_batch_size)

        # dynamic read/write
        sql_writer.write(
            [
                Experience(
                    tokens=torch.tensor([float(j) for j in range(i + 1)]),
                    reward=float(i),
                    logprobs=torch.tensor([0.1]),
                    action_mask=torch.tensor([j % 2 for j in range(i + 1)]),
                    info={"model_version": i + put_batch_size},
                )
                for i in range(1, put_batch_size * 2 + 1)
            ]
        )
        exps = sql_reader.read(batch_size=put_batch_size * 2)
        self.assertEqual(len(exps), put_batch_size * 2)
        for exp in exps:
            self.assertTrue(exp.info["model_version"] > put_batch_size)
        if enable_replay:
            # support replay, so we can read all again
            exps = sql_reader.read(batch_size=(put_batch_size * 2 + total_num))
            self.assertEqual(len(exps), (put_batch_size * 2 + total_num))
            # if read more than available, will wait until timeout
            with self.assertRaises(StopIteration):
                exps = sql_reader.read(batch_size=(put_batch_size * 3 + total_num))
        db_wrapper = ray.get_actor("sql-test_buffer")
        self.assertIsNotNone(db_wrapper)
        self.assertEqual(await sql_writer.release(), 0)
        self.assertRaises(StopIteration, sql_reader.read)

    async def test_sql_task_buffer_read_write(self) -> None:
        total_samples = 8
        batch_size = 4
        config = TasksetConfig(
            name="test_task_buffer",
            path=f"sqlite:///{db_path}",
            storage_type=StorageType.SQL.value,
            batch_size=batch_size,
            default_workflow_type="math_workflow",
        )
        sql_writer = SQLWriter(config.to_storage_config())
        tasks = [
            {"question": f"question_{i}", "answer": f"answer_{i}"} for i in range(total_samples)
        ]
        self.assertEqual(await sql_writer.acquire(), 1)
        sql_writer.write(tasks)
        sql_reader = get_buffer_reader(config.to_storage_config())
        read_tasks = []
        try:
            while True:
                cur_tasks = sql_reader.read()
                read_tasks.extend(cur_tasks)
        except StopIteration:
            pass
        self.assertEqual(len(read_tasks), total_samples)
        self.assertIn("question", read_tasks[0].raw_task)
        self.assertIn("answer", read_tasks[0].raw_task)
        db_wrapper = ray.get_actor("sql-test_task_buffer")
        self.assertIsNotNone(db_wrapper)
        self.assertEqual(await sql_writer.release(), 0)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)

    def tearDown(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)
