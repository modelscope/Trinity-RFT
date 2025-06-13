import unittest

from tests.tools import get_template_config, get_unittest_dataset_config
from trinity.buffer.buffer import get_buffer_reader


class TestFileReader(unittest.TestCase):
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
