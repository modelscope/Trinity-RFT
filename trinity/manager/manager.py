# -*- coding: utf-8 -*-
"""Data manager."""
import json
import os

from trinity.common.config import Config, load_config
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class CacheManager:
    """A Manager class for managing the cache dir."""

    def __init__(self, config: Config, check_config: bool = False):
        self.cache_dir = config.monitor.cache_dir  # type: ignore
        self.explorer_meta_path = os.path.join(self.cache_dir, f"{config.explorer.name}_meta.json")  # type: ignore
        self.trainer_meta_path = os.path.join(self.cache_dir, f"{config.trainer.name}_meta.json")  # type: ignore
        if check_config:
            self._check_config_consistency(config)

    def _check_config_consistency(self, config: Config) -> None:
        """Check if the config is consistent with the cache dir backup."""
        backup_config_path = os.path.join(self.cache_dir, "config.json")  # type: ignore
        if not os.path.exists(backup_config_path):
            config.save(backup_config_path)
        else:
            backup_config = load_config(backup_config_path)
            if backup_config != config:
                logger.warning(
                    f"The current config is inconsistent with the backup config in {backup_config_path}."
                )
                raise ValueError(
                    f"The current config is inconsistent with the backup config in {backup_config_path}."
                )

    def save_explorer(self, current_task_index: int, current_step: int) -> None:
        with open(self.explorer_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"latest_task_index": current_task_index, "latest_iteration": current_step},
                f,
                indent=2,
            )

    def load_explorer(self) -> dict:
        if os.path.exists(self.explorer_meta_path):
            try:
                with open(self.explorer_meta_path, "r", encoding="utf-8") as f:
                    explorer_meta = json.load(f)
                logger.info(
                    "----------------------------------\n"
                    "Found existing explorer checkpoint:\n"
                    f"  > {explorer_meta}\n"
                    "Continue exploring from this point.\n"
                    "----------------------------------"
                )
                return explorer_meta
            except Exception as e:
                logger.error(f"Failed to load explore meta file: {e}")
        return {}

    def save_trainer(self, current_step: int) -> None:
        with open(self.trainer_meta_path, "w", encoding="utf-8") as f:
            json.dump({"latest_iteration": current_step}, f, indent=2)

    def load_trainer(self) -> dict:
        if os.path.exists(self.trainer_meta_path):
            try:
                with open(self.trainer_meta_path, "r", encoding="utf-8") as f:
                    trainer_meta = json.load(f)
                logger.info(
                    "----------------------------------\n"
                    "Found existing trainer checkpoint:\n"
                    f"  > {trainer_meta}\n"
                    "Continue training from this point.\n"
                    "----------------------------------"
                )
                return trainer_meta
            except Exception as e:
                logger.warning(f"Failed to load trainer meta file: {e}")
        return {}
