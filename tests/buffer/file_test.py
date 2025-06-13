import os

from tests.tools import RayUnittestBase
from trinity.buffer.reader.file_reader import RawDataReader
from trinity.buffer.writer.file_writer import RawFileWriter
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class TestFileBuffer(RayUnittestBase):

    temp_output_path = 'tmp/test_file_buffer/'

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.temp_output_path, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if os.path.exists(cls.temp_output_path):
            os.system(f'rm -rf {cls.temp_output_path}')

    def test_file_buffer(self):
        meta = StorageConfig(
            name="test_buffer",
            path=os.path.join(self.temp_output_path, "buffer.jsonl"),
            storage_type=StorageType.FILE,
            raw=True,
        )
        data = [
            {'key1': 1, 'key2': 2},
            {'key1': 3, 'key2': 4},
            {'key1': 5, 'key2': 6},
            {'key1': 7, 'key2': 8},
        ]

        # test writer
        writer = RawFileWriter(meta, None)
        writer.write(data)
        writer.finish()

        # test reader
        meta.path = self.temp_output_path
        reader = RawDataReader(meta, None)
        loaded_data = reader.read()
        self.assertEqual(len(loaded_data), 4)
        self.assertEqual(loaded_data, data)
        self.assertRaises(StopIteration, reader.read)
