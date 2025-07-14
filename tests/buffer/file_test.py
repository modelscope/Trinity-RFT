import os
import unittest

import ray

from tests.tools import (
    get_checkpoint_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.buffer.buffer import get_buffer_reader, get_buffer_writer
from trinity.buffer.reader.file_reader import RawDataReader
from trinity.buffer.utils import default_storage_path
from trinity.buffer.writer.file_writer import JSONWriter
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class TestFileBuffer(unittest.IsolatedAsyncioTestCase):
    temp_output_path = "tmp/test_file_buffer/"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.makedirs(cls.temp_output_path, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if os.path.exists(cls.temp_output_path):
            os.system(f"rm -rf {cls.temp_output_path}")

    async def test_file_buffer(self):
        meta = StorageConfig(
            name="test_buffer",
            path=os.path.join(self.temp_output_path, "buffer.jsonl"),
            storage_type=StorageType.FILE,
            raw=True,
        )
        data = [
            {"key1": 1, "key2": 2},
            {"key1": 3, "key2": 4},
            {"key1": 5, "key2": 6},
            {"key1": 7, "key2": 8},
        ]

        # test writer
        writer = JSONWriter(meta, None)
        await writer.acquire()
        writer.write(data)
        await writer.release()

        # test reader
        meta.path = self.temp_output_path
        reader = RawDataReader(meta, None)
        loaded_data = reader.read()
        self.assertEqual(len(loaded_data), 4)
        self.assertEqual(loaded_data, data)
        self.assertRaises(StopIteration, reader.read)

    def test_file_reader(self):  # noqa: C901
        """Test file reader."""
        reader = get_buffer_reader(self.config.buffer.explorer_input.taskset, self.config.buffer)

        tasks = []
        while True:
            try:
                tasks.extend(reader.read())
            except StopIteration:
                break
        self.assertEqual(len(tasks), 16)

        # test epoch and offset
        self.config.buffer.explorer_input.taskset.total_epochs = 2
        self.config.buffer.explorer_input.taskset.index = 4
        reader = get_buffer_reader(self.config.buffer.explorer_input.taskset, self.config.buffer)
        tasks = []
        while True:
            try:
                tasks.extend(reader.read())
            except StopIteration:
                break
        self.assertEqual(len(tasks), 16 * 2 - 4)

        # test total steps and offset
        self.config.buffer.explorer_input.taskset.total_steps = 5
        self.config.buffer.explorer_input.taskset.index = 8
        reader = get_buffer_reader(self.config.buffer.explorer_input.taskset, self.config.buffer)
        tasks = []
        while True:
            try:
                tasks.extend(reader.read())
                print(f"read from buffer, current len {len(tasks)}.")
            except StopIteration:
                break
        self.assertEqual(len(tasks), 20 - 8)

        # test offset > dataset_len with total_epoch
        self.config.buffer.explorer_input.taskset.total_steps = None
        self.config.buffer.explorer_input.taskset.total_epochs = 3
        self.config.buffer.explorer_input.taskset.index = 20
        reader = get_buffer_reader(self.config.buffer.explorer_input.taskset, self.config.buffer)
        tasks = []
        while True:
            try:
                tasks.extend(reader.read())
            except StopIteration:
                break
        self.assertEqual(len(tasks), 16 * 3 - 20)

        # test offset > dataset_len with total_steps
        self.config.buffer.explorer_input.taskset.total_steps = 10
        self.config.buffer.explorer_input.taskset.index = 24
        reader = get_buffer_reader(self.config.buffer.explorer_input.taskset, self.config.buffer)
        tasks = []
        while True:
            try:
                tasks.extend(reader.read())
            except StopIteration:
                break
        self.assertEqual(len(tasks), 40 - 24)

    async def test_file_writer(self):
        writer = get_buffer_writer(
            self.config.buffer.trainer_input.experience_buffer, self.config.buffer
        )
        await writer.acquire()
        writer.write(
            [
                {"prompt": "hello world"},
                {"prompt": "hi"},
            ]
        )
        await writer.write_async(
            [
                {"prompt": "My name is"},
                {"prompt": "What is your name?"},
            ]
        )
        await writer.release()
        file_wrapper = ray.get_actor("json-test_buffer")
        self.assertIsNotNone(file_wrapper)
        file_path = default_storage_path(
            self.config.buffer.trainer_input.experience_buffer, self.config.buffer
        )
        with open(file_path, "r") as f:
            self.assertEqual(len(f.readlines()), 4)

    def setUp(self):
        self.config = get_template_config()
        self.config.checkpoint_root_dir = get_checkpoint_path()
        dataset_config = get_unittest_dataset_config("countdown", "train")
        self.config.buffer.explorer_input.taskset = dataset_config
        self.config.buffer.trainer_input.experience_buffer = StorageConfig(
            name="test_buffer", storage_type=StorageType.FILE
        )
        self.config.buffer.trainer_input.experience_buffer.name = "test_buffer"
        self.config.buffer.cache_dir = os.path.join(
            self.config.checkpoint_root_dir, self.config.project, self.config.name, "buffer"
        )
        os.makedirs(self.config.buffer.cache_dir, exist_ok=True)
        if os.path.exists(
            default_storage_path(
                self.config.buffer.trainer_input.experience_buffer, self.config.buffer
            )
        ):
            os.remove(
                default_storage_path(
                    self.config.buffer.trainer_input.experience_buffer, self.config.buffer
                )
            )
