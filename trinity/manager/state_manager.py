# -*- coding: utf-8 -*-
"""State manager."""
import json
import os
from typing import Dict, List, Optional

from trinity.common.config import Config, load_config
from trinity.utils.log import get_logger


class StateManager:
    """A Manager class for managing the running state of Explorer and Trainer."""

    def __init__(
        self,
        path: str,
        trainer_name: Optional[str] = None,
        explorer_name: Optional[str] = None,
        config: Optional[Config] = None,
        check_config: bool = False,
    ):
        self.logger = get_logger(__name__, in_ray_actor=True)
        self.cache_dir = path
        os.makedirs(self.cache_dir, exist_ok=True)
        self.stage_state_path = os.path.join(self.cache_dir, "stage_meta.json")
        self.explorer_state_path = os.path.join(self.cache_dir, f"{explorer_name}_meta.json")
        self.trainer_state_path = os.path.join(self.cache_dir, f"{trainer_name}_meta.json")
        self.explorer_server_url_path = os.path.join(
            self.cache_dir, f"{explorer_name}_server_url.txt"
        )
        if check_config and config is not None:
            self._check_config_consistency(config)

    def _check_config_consistency(self, config: Config) -> None:
        """Check if the config is consistent with the cache dir backup."""
        backup_config_path = os.path.join(self.cache_dir, "config.json")
        if not os.path.exists(backup_config_path):
            config.save(backup_config_path)
        else:
            backup_config = load_config(backup_config_path)
            if backup_config != config:
                self.logger.warning(
                    f"The current config is inconsistent with the backup config in {backup_config_path}."
                )
                raise ValueError(
                    f"The current config is inconsistent with the backup config in {backup_config_path}."
                )

    def save_explorer(
        self,
        current_step: int,
        taskset_states: List[Dict],
    ) -> None:
        with open(self.explorer_state_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "latest_iteration": current_step,
                    "taskset_states": taskset_states,
                },
                f,
                indent=2,
            )

    def load_explorer(self) -> dict:
        if os.path.exists(self.explorer_state_path):
            try:
                with open(self.explorer_state_path, "r", encoding="utf-8") as f:
                    explorer_meta = json.load(f)
                self.logger.info(
                    "----------------------------------\n"
                    "Found existing explorer checkpoint:\n"
                    f"  > {explorer_meta}\n"
                    "Continue exploring from this point.\n"
                    "----------------------------------"
                )
                return explorer_meta
            except Exception as e:
                self.logger.error(f"Failed to load explore state file: {e}")
        return {}

    def save_explorer_server_url(self, url: str) -> None:
        with open(self.explorer_server_url_path, "w", encoding="utf-8") as f:
            f.write(url)
        self.logger.info(f"Saved explorer server URL to {self.explorer_server_url_path}")

    def load_explorer_server_url(self) -> Optional[str]:
        if os.path.exists(self.explorer_server_url_path):
            try:
                with open(self.explorer_server_url_path, "r", encoding="utf-8") as f:
                    url = f.read().strip()
                self.logger.info(
                    "----------------------------------\n"
                    "Found existing explorer server URL:\n"
                    f"  > {url}\n"
                    "----------------------------------"
                )
                return url
            except Exception as e:
                self.logger.error(f"Failed to load explorer server URL file: {e}")
        return None

    def save_trainer(
        self,
        current_exp_index: int,
        current_step: int,
    ) -> None:
        with open(self.trainer_state_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "latest_exp_index": current_exp_index,
                    "latest_iteration": current_step,
                },
                f,
                indent=2,
            )

    def load_trainer(self) -> dict:
        if os.path.exists(self.trainer_state_path):
            try:
                with open(self.trainer_state_path, "r", encoding="utf-8") as f:
                    trainer_meta = json.load(f)
                self.logger.info(
                    "----------------------------------\n"
                    "Found existing trainer checkpoint:\n"
                    f"  > {trainer_meta}\n"
                    "Continue training from this point.\n"
                    "----------------------------------"
                )
                return trainer_meta
            except Exception as e:
                self.logger.warning(f"Failed to load trainer state file: {e}")
        return {}

    def save_stage(self, current_stage: int) -> None:
        with open(self.stage_state_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "latest_stage": current_stage,
                },
                f,
                indent=2,
            )

    def load_stage(self) -> dict:
        if os.path.exists(self.stage_state_path):
            try:
                with open(self.stage_state_path, "r", encoding="utf-8") as f:
                    stage_meta = json.load(f)
                self.logger.info(
                    "----------------------------------\n"
                    "Found existing stage checkpoint:\n"
                    f"  > {stage_meta}\n"
                    "Continue from this point.\n"
                    "----------------------------------"
                )
                return stage_meta
            except Exception as e:
                self.logger.warning(f"Failed to load stage state file: {e}")
        return {}
