import os
import time

import torch

from tests.tools import RayUnittestBase
from trinity.buffer.reader.queue_reader import QueueReader
from trinity.buffer.writer.queue_writer import QueueWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience

BUFFER_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_queue_buffer.jsonl")


class TestQueueBuffer(RayUnittestBase):
    def test_queue_buffer(self):
        total_num = 8
        put_batch_size = 2
        read_batch_size = 4
        meta = StorageConfig(
            name="test_buffer",
            algorithm_type="ppo",
            storage_type=StorageType.QUEUE,
            max_read_timeout=3,
            path=BUFFER_FILE_PATH,
        )
        config = BufferConfig(
            max_retry_times=3,
            max_retry_interval=1,
            read_batch_size=read_batch_size,
        )
        writer = QueueWriter(meta, config)
        reader = QueueReader(meta, config)
        self.assertEqual(writer.acquire(), 1)
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                prompt_length=i,
                reward=float(i),
                logprobs=torch.tensor([0.1]),
            )
            for i in range(1, put_batch_size + 1)
        ]
        for _ in range(total_num // put_batch_size):
            writer.write(exps)
        for _ in range(total_num // read_batch_size):
            exps = reader.read()
            self.assertEqual(len(exps), read_batch_size)
            print(f"finish read {read_batch_size} experience")
        writer.write(
            [
                Experience(
                    tokens=torch.tensor([float(j) for j in range(i + 1)]),
                    prompt_length=i,
                    reward=float(i),
                    logprobs=torch.tensor([0.1]),
                    action_mask=torch.tensor([j % 2 for j in range(i + 1)]),
                )
                for i in range(1, put_batch_size * 2 + 1)
            ]
        )
        exps = reader.read(batch_size=put_batch_size * 2)
        self.assertEqual(len(exps), put_batch_size * 2)
        self.assertEqual(writer.release(), 0)
        self.assertRaises(StopIteration, reader.read)
        with open(BUFFER_FILE_PATH, "r") as f:
            self.assertEqual(len(f.readlines()), total_num + put_batch_size * 2)
        st = time.time()
        self.assertRaises(StopIteration, reader.read, batch_size=1)
        et = time.time()
        self.assertTrue(et - st > 2)

    def setUp(self):
        if os.path.exists(BUFFER_FILE_PATH):
            os.remove(BUFFER_FILE_PATH)
