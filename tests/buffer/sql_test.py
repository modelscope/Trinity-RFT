import os
import unittest

import torch

from trinity.buffer.reader.sql_reader import SQLReader
from trinity.buffer.writer.sql_writer import SQLWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import AlgorithmType, StorageType
from trinity.common.experience import Experience

db_path = os.path.join(os.path.dirname(__file__), "test.db")


class TestSQLBuffer(unittest.TestCase):
    def test_create_sql_buffer(self) -> None:
        total_num = 8
        put_batch_size = 2
        read_batch_size = 4
        meta = StorageConfig(
            name="test_buffer",
            algorithm_type=AlgorithmType.PPO,
            path=f"sqlite:///{db_path}",
            storage_type=StorageType.SQL,
        )
        config = BufferConfig(
            max_retry_times=3,
            max_retry_interval=1,
            read_batch_size=read_batch_size,
        )
        sql_writer = SQLWriter(meta, config)
        sql_reader = SQLReader(meta, config)
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                prompt_length=i,
                reward=float(i),
                logprobs=torch.tensor([0.1]),
                action_mask=torch.tensor([j % 2 for j in range(i + 1)]),
            )
            for i in range(1, put_batch_size + 1)
        ]
        for _ in range(total_num // put_batch_size):
            sql_writer.write(exps)
        for _ in range(total_num // read_batch_size):
            exps = sql_reader.read()
            self.assertEqual(len(exps), read_batch_size)
