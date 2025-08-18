import os
import unittest
from pathlib import Path

import ray

from tests.tools import TensorBoardParser, get_template_config
from trinity.common.config import Config
from trinity.common.workflows import WORKFLOWS
from trinity.utils.monitor import MONITOR
from trinity.utils.plugin_loader import load_plugins


class PluginActor:
    def __init__(self, config: Config):
        self.config = config
        self.monitor = MONITOR.get("my_monitor")(
            project=self.config.project,
            group=self.config.group,
            name=self.config.name,
            role=self.config.explorer.name,
            config=config,
        )

    def run(self):
        my_plugin_cls = WORKFLOWS.get("my_workflow")
        self.monitor.log({"rollout": 2}, step=1, commit=True)
        return my_plugin_cls(task=None, model=None).run()


class TestPluginLoader(unittest.TestCase):
    def test_load_plugins(self):
        ray.init(ignore_reinit_error=True)
        config = get_template_config()
        my_plugin_cls = WORKFLOWS.get("my_workflow")
        self.assertIsNone(my_plugin_cls)
        load_plugins(Path(__file__).resolve().parent / "plugins")
        my_plugin_cls = WORKFLOWS.get("my_workflow")
        self.assertIsNotNone(my_plugin_cls)
        my_plugin = my_plugin_cls(task=None, model=None, auxiliary_models=None)
        self.assertTrue(my_plugin.__module__.startswith("trinity.plugins"))
        res = my_plugin.run()
        self.assertEqual(res[0], "Hello world")
        self.assertEqual(res[1], "Hi")

        # Remote Actor test
        remote_plugin = ray.remote(PluginActor).remote(config)
        remote_res = ray.get(remote_plugin.run.remote())
        self.assertEqual(remote_res[0], "Hello world")
        self.assertEqual(remote_res[1], "Hi")

        # test custom monitor
        parser = TensorBoardParser(os.path.join(config.monitor.cache_dir, "tensorboard"))
        rollout_cnt = parser.metric_values("rollout")
        self.assertEqual(rollout_cnt, [2])
        ray.shutdown(_exiting_interpreter=True)
