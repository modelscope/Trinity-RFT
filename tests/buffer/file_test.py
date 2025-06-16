import os
import unittest

from tests.tools import get_template_config, get_unittest_dataset_config
from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.reader.file_reader import RawDataReader
from trinity.buffer.writer.file_writer import JSONWriter
from trinity.common.config import StorageConfig, StorageType


class TestFileReader(unittest.TestCase):
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

    def test_file_buffer(self):
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
        writer.write(data)
        writer.finish()

        # test reader
        meta.path = self.temp_output_path
        reader = RawDataReader(meta, None)
        loaded_data = reader.read()
        self.assertEqual(len(loaded_data), 4)
        self.assertEqual(loaded_data, data)
        self.assertRaises(StopIteration, reader.read)

    def test_file_reader(self):
        """Test file reader."""
        config = get_template_config()
        dataset_config = get_unittest_dataset_config("countdown", "train")
        config.buffer.explorer_input.taskset = dataset_config
        reader = get_buffer_reader(config.buffer.explorer_input.taskset, config.buffer)

        tasks = []
        while True:
            try:
                tasks.extend(reader.read())
            except StopIteration:
                break
        self.assertEqual(len(tasks), 16)

        config.buffer.explorer_input.taskset.total_epochs = 2
        config.buffer.explorer_input.taskset.index = 4
        reader = get_buffer_reader(config.buffer.explorer_input.taskset, config.buffer)
        tasks = []
        while True:
            try:
                tasks.extend(reader.read())
            except StopIteration:
                break
        self.assertEqual(len(tasks), 16 * 2 - 4)
