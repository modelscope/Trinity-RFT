import os
import time
from contextlib import contextmanager

from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.utils.log import get_logger


@contextmanager
def retry_session(session_maker, max_retry_times: int, max_retry_interval: float):
    """A Context manager for retrying session."""
    logger = get_logger(__name__)
    for attempt in range(max_retry_times):
        try:
            session = session_maker()
            yield session
            session.commit()
            break
        except StopIteration as e:
            raise e
        except Exception as e:
            import traceback

            trace_str = traceback.format_exc()
            session.rollback()
            logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {max_retry_interval} seconds..."
            )
            logger.warning(f"trace = {trace_str}")
            if attempt < max_retry_times - 1:
                time.sleep(max_retry_interval)
            else:
                logger.error("Max retry attempts reached, raising exception.")
                raise e
        finally:
            session.close()


def default_storage_path(storage_config: StorageConfig, buffer_config: BufferConfig) -> str:
    if buffer_config.cache_dir is None:
        raise ValueError("Please call config.check_and_update() before using.")
    if storage_config.storage_type == StorageType.SQL:
        return "sqlite:///" + os.path.join(
            buffer_config.cache_dir,
            f"{storage_config.name}.db",
        )
    else:
        return os.path.join(
            buffer_config.cache_dir,
            f"{storage_config.name}.jsonl",
        )
