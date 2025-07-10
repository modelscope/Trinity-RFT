"""Reader of the Replay Buffer."""

from trinity.buffer.reader.queue_reader import QueueReader
from trinity.buffer.replay_buffer import ReplayBufferActor
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class ReplayBufferReader(QueueReader):
    """Reader of the Replay Buffer."""

    def __init__(self, storage_config: StorageConfig, config: BufferConfig):
        assert storage_config.storage_type == StorageType.REPLAY_BUFFER
        self.timeout = storage_config.max_read_timeout
        self.read_batch_size = config.read_batch_size
        self.queue = ReplayBufferActor.get_actor(storage_config, config)
