import torch

from tests.tools import RayUnittestBase
from trinity.buffer.reader.queue_reader import QueueReader
from trinity.buffer.writer.queue_writer import QueueWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import AlgorithmType, StorageType
from trinity.common.experience import Experience


class TestQueueBuffer(RayUnittestBase):
    def test_queue_buffer(self):
        total_num = 8
        put_batch_size = 2
        read_batch_size = 4
        meta = StorageConfig(
            name="test_buffer",
            namespace="test_namespace",
            algorithm_type=AlgorithmType.PPO,
            storage_type=StorageType.QUEUE,
        )
        config = BufferConfig(
            max_retry_times=3,
            max_retry_interval=1,
            read_batch_size=read_batch_size,
        )
        writer = QueueWriter(meta, config)
        reader = QueueReader(meta, config)
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
        writer.finish()
        for _ in range(total_num // read_batch_size):
            exps = reader.read()
            self.assertEqual(len(exps), read_batch_size)
            print(f"finish read {read_batch_size} experience")
        self.assertRaises(StopIteration, reader.read)
