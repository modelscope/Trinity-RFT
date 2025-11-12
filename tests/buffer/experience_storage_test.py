import os
import queue
import sqlite3
import threading
import time

import torch
from parameterized import parameterized

from tests.tools import RayUnittestBaseAysnc
from trinity.buffer.reader.sql_reader import SQLReader
from trinity.buffer.writer.sql_writer import SQLWriter
from trinity.common.config import ExperienceBufferConfig
from trinity.common.constants import StorageType
from trinity.common.experience import EID, Experience

DB_PATH = os.path.join(os.path.dirname(__file__), "test.db")


class ExperienceStorageTest(RayUnittestBaseAysnc):
    def setUp(self):
        self.total_num = 8
        self.put_batch_size = 2
        self.train_batch_size = 4

        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

    @parameterized.expand([("sft",), ("dpo",)])
    async def test_sql_storage(self, schema_type):
        config = ExperienceBufferConfig(
            name="test_storage",
            schema_type=schema_type,
            storage_type=StorageType.SQL,
            max_read_timeout=3,
            path=f"sqlite:///{DB_PATH}",
            batch_size=self.train_batch_size,
        )
        config = config.to_storage_config()
        writer = SQLWriter(config)
        reader = SQLReader(config)
        self.assertEqual(await writer.acquire(), 1)
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                prompt_length=i,
                reward=float(i),
                logprobs=torch.tensor([0.1]),
            )
            for i in range(1, self.put_batch_size + 1)
        ]
        for _ in range(self.total_num // self.put_batch_size):
            await writer.write_async(exps)
        for _ in range(self.total_num // self.train_batch_size):
            exps = reader.read()
            self.assertEqual(len(exps), self.train_batch_size)
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                reward=float(i),
                logprobs=torch.tensor([0.1]),
                action_mask=torch.tensor([j % 2 for j in range(i + 1)]),
            )
            for i in range(1, self.put_batch_size * 2 + 1)
        ]
        writer.write(exps)
        exps = reader.read(batch_size=self.put_batch_size * 2)
        self.assertEqual(len(exps), self.put_batch_size * 2)

        def thread_read(reader, result_queue):
            try:
                batch = reader.read()
                result_queue.put(batch)
            except StopIteration as e:
                result_queue.put(e)

        result_queue = queue.Queue()
        t = threading.Thread(target=thread_read, args=(reader, result_queue))
        t.start()
        time.sleep(2)  # make sure the thread is waiting for data
        self.assertEqual(await writer.release(), 0)
        t.join(timeout=1)
        self.assertIsInstance(result_queue.get(), StopIteration)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        value = cursor.execute("SELECT COUNT(*) FROM test_storage;").fetchall()
        self.assertEqual(value[0][0], self.total_num + self.put_batch_size * 2)
        self.assertRaises(StopIteration, reader.read, batch_size=1)

    async def test_sql_experience_buffer(self):
        config = ExperienceBufferConfig(
            name="test_storage",
            schema_type="experience",
            storage_type=StorageType.SQL,
            max_read_timeout=3,
            path=f"sqlite:///{DB_PATH}",
            batch_size=self.train_batch_size,
        )
        config = config.to_storage_config()
        writer = SQLWriter(config)
        reader = SQLReader(config)
        self.assertEqual(await writer.acquire(), 1)
        for idx in range(self.total_num // self.put_batch_size):
            exps = [
                Experience(
                    eid=EID(task=idx * self.put_batch_size + i),
                    tokens=torch.tensor([float(j) for j in range(i + 1)]),
                    prompt_length=i,
                    reward=float(i),
                    logprobs=torch.tensor([0.1]),
                )
                for i in range(1, self.put_batch_size + 1)
            ]
            await writer.write_async(exps)
        cnt = self.total_num
        for _ in range(self.total_num // self.train_batch_size):
            exps = reader.read()
            self.assertEqual(len(exps), self.train_batch_size)
            for exp in exps:
                self.assertEqual(exp.eid.task, cnt)
                cnt -= 1

        # experience buffer support experience reuse
        cnt = self.total_num
        for _ in range(self.total_num // self.train_batch_size):
            exps = reader.read()
            self.assertEqual(len(exps), self.train_batch_size)
            for exp in exps:
                self.assertEqual(exp.eid.task, cnt)
                cnt -= 1
        self.assertEqual(await writer.release(), 0)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        value = cursor.execute("SELECT COUNT(*) FROM test_storage;").fetchall()
        self.assertEqual(value[0][0], self.total_num)
