import os
import threading
import time

import torch
from parameterized import parameterized

from tests.tools import RayUnittestBaseAysnc
from trinity.buffer.reader.queue_reader import QueueReader
from trinity.buffer.writer.queue_writer import QueueWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience

BUFFER_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_queue_buffer.jsonl")


class TestQueueBuffer(RayUnittestBaseAysnc):
    @parameterized.expand(
        [
            (
                "queue",
                False,
            ),
            (
                "priority_queue",
                True,
            ),
        ]
    )
    async def test_queue_buffer(self, name, use_priority_queue):
        meta = StorageConfig(
            name="test_buffer",
            algorithm_type="ppo",
            storage_type=StorageType.QUEUE,
            max_read_timeout=3,
            path=BUFFER_FILE_PATH,
            use_priority_queue=use_priority_queue,
        )
        writer = QueueWriter(meta, self.config)
        reader = QueueReader(meta, self.config)
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
        for exp in exps:
            exp.info = {"model_version": 0, "use_count": 0}
        for _ in range(self.total_num // self.put_batch_size):
            await writer.write_async(exps)
        for _ in range(self.total_num // self.read_batch_size):
            exps = reader.read()
            self.assertEqual(len(exps), self.read_batch_size)
            print(f"finish read {self.read_batch_size} experience")
        exps = [
            Experience(
                tokens=torch.tensor([float(j) for j in range(i + 1)]),
                prompt_length=i,
                reward=float(i),
                logprobs=torch.tensor([0.1]),
                action_mask=torch.tensor([j % 2 for j in range(i + 1)]),
            )
            for i in range(1, self.put_batch_size * 2 + 1)
        ]
        for exp in exps:
            exp.info = {"model_version": 1, "use_count": 0}
        writer.write(exps)
        exps = reader.read(batch_size=self.put_batch_size * 2)
        self.assertEqual(len(exps), self.put_batch_size * 2)
        self.assertEqual(await writer.release(), 0)
        self.assertRaises(StopIteration, reader.read)
        with open(BUFFER_FILE_PATH, "r") as f:
            self.assertEqual(len(f.readlines()), self.total_num + self.put_batch_size * 2)
        st = time.time()
        self.assertRaises(TimeoutError, reader.read, batch_size=1)
        et = time.time()
        self.assertTrue(et - st > 2)

    async def test_priority_queue_capacity(self):
        # test queue capacity
        meta = StorageConfig(
            name="test_buffer_small",
            algorithm_type="ppo",
            storage_type=StorageType.QUEUE,
            max_read_timeout=1,
            capacity=2,
            path=BUFFER_FILE_PATH,
            use_priority_queue=True,
            replay_buffer_kwargs={"priority_fn": "linear_decay", "decay": 0.6},
        )
        writer = QueueWriter(meta, self.config)
        reader = QueueReader(meta, self.config)

        for i in range(4):
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                ]
            )

        exps = reader.read(batch_size=2)
        self.assertEqual(exps[0].info["model_version"], 3)
        self.assertEqual(exps[0].info["use_count"], 1)
        self.assertEqual(exps[1].info["model_version"], 2)
        self.assertEqual(exps[1].info["use_count"], 1)

        with self.assertRaises(TimeoutError):
            reader.read(batch_size=1)

    async def test_queue_buffer_capacity(self):
        # test queue capacity
        meta = StorageConfig(
            name="test_buffer_small",
            algorithm_type="ppo",
            storage_type=StorageType.QUEUE,
            max_read_timeout=3,
            capacity=4,
            path=BUFFER_FILE_PATH,
        )
        writer = QueueWriter(meta, self.config)
        reader = QueueReader(meta, self.config)
        writer.write([{"content": "hello"}])
        writer.write([{"content": "hi"}])
        writer.write([{"content": "hello"}])
        writer.write([{"content": "hi"}])

        # should be blocked
        def write_blocking_call():
            writer.write([{"content": "blocked"}])

        thread = threading.Thread(target=write_blocking_call)
        thread.start()
        thread.join(timeout=2)
        self.assertTrue(thread.is_alive(), "write() did not block as expected")
        reader.read()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

    async def test_priority_queue_buffer_reuse(self):
        # test queue reuse
        meta = StorageConfig(
            name="test_buffer_small",
            algorithm_type="ppo",
            storage_type=StorageType.QUEUE,
            max_read_timeout=3,
            capacity=4,
            path=BUFFER_FILE_PATH,
            use_priority_queue=True,
            reuse_cooldown_time=0.5,
            replay_buffer_kwargs={"priority_fn": "linear_decay", "decay": 0.6},
        )
        writer = QueueWriter(meta, self.config)
        reader = QueueReader(meta, self.config)
        for i in range(4):
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": i, "use_count": 0},
                    ),
                ]
            )

        # should not be blocked
        def replace_call():
            writer.write(
                [
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": 4, "use_count": 0},
                    ),
                    Experience(
                        tokens=torch.tensor([1, 2, 3]),
                        prompt_length=2,
                        info={"model_version": 4, "use_count": 0},
                    ),
                ]
            )

        thread = threading.Thread(target=replace_call)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 1)
        self.assertEqual(exps[2].info["model_version"], 3)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  4,   3,   2,   1
        # use_count      1,   1,   0,   0
        # priority      3.4, 2.4, 2.0, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 2)
        self.assertEqual(exps[2].info["model_version"], 3)
        self.assertEqual(exps[2].info["use_count"], 2)

        # model_version  4,   3,   2,   1
        # use_count      2,   2,   0,   0
        # priority      2.8, 1.8, 2.0, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 3)
        self.assertEqual(exps[2].info["model_version"], 2)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  4,   3,   2,   1
        # use_count      3,   2,   1,   0
        # priority      2.2, 1.8, 1.4, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 4)
        self.assertEqual(exps[2].info["model_version"], 3)
        self.assertEqual(exps[2].info["use_count"], 3)

        # model_version  4,   3,   2,   1
        # use_count      4,   3,   1,   0
        # priority      1.6, 1.2, 1.4, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 4)
        self.assertEqual(exps[0].info["use_count"], 5)
        self.assertEqual(exps[2].info["model_version"], 2)
        self.assertEqual(exps[2].info["use_count"], 2)

        # model_version  4,   3,   2,   1
        # use_count      5,   3,   2,   0
        # priority      1.0, 1.2, 0.8, 1.0

        time.sleep(1)
        exps = reader.read(batch_size=4)
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].info["model_version"], 3)
        self.assertEqual(exps[0].info["use_count"], 4)
        self.assertEqual(exps[2].info["model_version"], 1)
        self.assertEqual(exps[2].info["use_count"], 1)

        # model_version  4,   3,   2,   1
        # use_count      5,   4,   2,   1
        # priority      1.0, 0.6, 0.8, 0.4

    def setUp(self):
        self.total_num = 8
        self.put_batch_size = 2
        self.read_batch_size = 4

        self.config = BufferConfig(
            max_retry_times=3,
            max_retry_interval=1,
            read_batch_size=self.read_batch_size,
        )
        if os.path.exists(BUFFER_FILE_PATH):
            os.remove(BUFFER_FILE_PATH)
