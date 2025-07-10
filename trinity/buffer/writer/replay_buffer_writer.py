"""Writer of the Replay Buffer."""
from trinity.buffer.replay_buffer import ReplayBufferActor
from trinity.buffer.writer.queue_writer import QueueWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class ReplayBufferWriter(QueueWriter):
    """Writer of the Replay Buffer."""

    def __init__(self, meta: StorageConfig, config: BufferConfig):
        assert meta.storage_type == StorageType.REPLAY_BUFFER
        self.config = config
        self.queue = ReplayBufferActor.get_actor(meta, config)
