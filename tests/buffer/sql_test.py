import os

import ray
import torch

from tests.tools import RayUnittestBaseAysnc
from trinity.buffer.reader.sql_reader import SQLReader
from trinity.buffer.writer.sql_writer import SQLWriter
from trinity.common.config import ExperienceBufferConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience

db_path = os.path.join(os.path.dirname(__file__), "test.db")


class TestSQLBuffer(RayUnittestBaseAysnc):
    async def test_sql_buffer_read_write(self) -> None:
        total_num = 8
        put_batch_size = 2
        read_batch_size = 4
        config = ExperienceBufferConfig(
            name="test_buffer",
            schema_type="experience",
            path=f"sqlite:///{db_path}",
            storage_type=StorageType.SQL,
            batch_size=read_batch_size,
        )
        sql_writer = SQLWriter(config.to_storage_config())
        sql_reader = SQLReader(config.to_storage_config())
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                prompt_length=i,
                reward=float(i),
                logprobs=torch.tensor([0.1]),
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
                )
                for i in range(1, put_batch_size * 2 + 1)
            ]
        )
        exps = sql_reader.read(batch_size=put_batch_size * 2)
        self.assertEqual(len(exps), put_batch_size * 2)
        db_wrapper = ray.get_actor("sql-test_buffer")
        self.assertIsNotNone(db_wrapper)
        self.assertEqual(await sql_writer.release(), 0)
        self.assertRaises(StopIteration, sql_reader.read)

    def setUp(self) -> None:
        if os.path.exists(db_path):
            os.remove(db_path)
