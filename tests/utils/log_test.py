import logging
import os
import shutil
import unittest

import ray
from ray.runtime_env import RuntimeEnv

from tests.tools import get_template_config
from trinity.common.constants import (
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    LOG_NODE_IP_ENV_VAR,
)
from trinity.utils.log import get_logger


def log_outside_actor(log_level=logging.INFO):
    logger = get_logger("outside_actor", level=log_level)
    logger.info("Outside logger initialized")
    logger.debug("Outside logger initialized")


class ModuleInActor:
    def __init__(self):
        self.logger = get_logger("module_in_actor", in_ray_actor=True)
        self.logger.info("ModuleInActor initialized")
        self.logger.debug("ModuleInActor initialized")


class ModuleInActor2:
    def __init__(self):
        # module create in actor should automatically inherit the logger created by the root actor
        self.logger = get_logger("module_in_actor2")
        self.logger.info("ModuleInActor2 initialized")
        self.logger.debug("ModuleInActor2 initialized")


@ray.remote
class ActorInActor:
    """An actor created inside an actor"""

    def __init__(self, parent_name, log_level):
        self.logger = get_logger(f"{parent_name}_nested", in_ray_actor=True, level=log_level)
        self.logger.info("ActorInActor initialized")
        self.logger.debug("ActorInActor initialized")


@ray.remote
class LogActor:
    def __init__(self, aid: int, log_level=logging.INFO):
        assert os.environ.get(LOG_DIR_ENV_VAR) is not None, "LOG_DIR_ENV_VAR must be set"
        self.logger = get_logger(f"actor_{aid}", in_ray_actor=True, level=log_level)
        self.logger.info(f"LogActor {aid} initialized ")
        self.logger.debug(f"LogActor {aid} initialized")
        self.aid = aid
        self.actor = ActorInActor.remote(f"actor_{aid}", log_level)
        ray.get(self.actor.__ray_ready__.remote())

    def log_info(self, message: str):
        self.logger.info(f"LogActor {self.aid} info: {message}")
        self.logger.debug(f"LogActor {self.aid} debug: {message}")
        ModuleInActor()
        ModuleInActor2()


class LogTest(unittest.TestCase):
    def setUp(self):
        if ray.is_initialized():
            ray.shutdown()
        self.config = get_template_config()
        self.config.check_and_update()
        self.log_dir = self.config.log.save_dir
        shutil.rmtree(self.log_dir, ignore_errors=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def test_no_actor_log(self):
        ray.init(
            namespace=self.config.ray_namespace,
            runtime_env=RuntimeEnv(
                env_vars={LOG_DIR_ENV_VAR: self.log_dir, LOG_LEVEL_ENV_VAR: "INFO"}
            ),
        )
        try:
            logger = get_logger("outside_actor", level=logging.DEBUG)
            logger.info("Outside logger initialized")
            logger.debug("Outside logger initialized")
            self.assertFalse(os.path.exists(os.path.join(self.log_dir, "outside_actor.log")))

            logger = get_logger(
                "outside_actor", in_ray_actor=True
            )  # in_ray_actor should not take effect
            logger.info("Outside logger initialized")
            self.assertFalse(os.path.exists(os.path.join(self.log_dir, "outside_actor.log")))

        finally:
            ray.shutdown(_exiting_interpreter=True)

    def test_actor_log(self):
        ray.init(
            namespace=self.config.ray_namespace,
            runtime_env=RuntimeEnv(
                env_vars={
                    LOG_DIR_ENV_VAR: self.log_dir,
                    LOG_LEVEL_ENV_VAR: "INFO",
                }
            ),
        )
        try:
            actor1 = LogActor.remote(1, log_level=logging.INFO)
            actor2 = LogActor.remote(2, log_level=logging.DEBUG)
            actor3 = LogActor.remote(3, log_level=None)
            ray.get(actor1.log_info.remote("Test message"))
            ray.get(actor2.log_info.remote("Test message"))
            ray.get(actor3.log_info.remote("Test message"))
            self.assertTrue(os.path.exists(os.path.join(self.log_dir, "actor_1.log")))
            self.assertTrue(os.path.exists(os.path.join(self.log_dir, "actor_2.log")))
            self.assertTrue(os.path.exists(os.path.join(self.log_dir, "actor_3.log")))
            self.assertTrue(os.path.exists(os.path.join(self.log_dir, "actor_1_nested.log")))
            self.assertTrue(os.path.exists(os.path.join(self.log_dir, "actor_2_nested.log")))
            self.assertTrue(os.path.exists(os.path.join(self.log_dir, "actor_3_nested.log")))
            self.assertFalse(os.path.exists(os.path.join(self.log_dir, "module_in_actor.log")))
            self.assertFalse(os.path.exists(os.path.join(self.log_dir, "module_in_actor2.log")))
            with open(os.path.join(self.log_dir, "actor_1.log"), "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 4)
            with open(os.path.join(self.log_dir, "actor_2.log"), "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 8)
            with open(os.path.join(self.log_dir, "actor_3.log"), "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 4)
            with open(os.path.join(self.log_dir, "actor_1_nested.log"), "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
            with open(os.path.join(self.log_dir, "actor_2_nested.log"), "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
            with open(os.path.join(self.log_dir, "actor_3_nested.log"), "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
        finally:
            ray.shutdown(_exiting_interpreter=True)

    def test_group_by_node(self):
        ray.init(
            namespace=self.config.ray_namespace,
            runtime_env=RuntimeEnv(
                env_vars={
                    LOG_DIR_ENV_VAR: self.log_dir,
                    LOG_LEVEL_ENV_VAR: "INFO",
                    LOG_NODE_IP_ENV_VAR: "1",
                }
            ),
        )
        try:
            actor = LogActor.remote(1, log_level=logging.INFO)
            ray.get(actor.log_info.remote("Test message"))
            ips = os.listdir(self.config.log.save_dir)
            self.assertTrue(len(ips) > 0)
            for ip in ips:
                self.assertTrue(os.path.isdir(os.path.join(self.config.log.save_dir, ip)))
                ip_logs = os.listdir(os.path.join(self.config.log.save_dir, ip))
                self.assertTrue(len(ip_logs) > 0)
        finally:
            ray.shutdown(_exiting_interpreter=True)
