import unittest
from pathlib import Path

import ray

from trinity.common.workflows import WORKFLOWS
from trinity.utils.plugin_loader import load_plugins


@ray.remote
class PluginActor:
    def run(self):
        my_plugin_cls = WORKFLOWS.get("my_workflow")
        return my_plugin_cls(task=None, model=None).run()


class TestPluginLoader(unittest.TestCase):
    def test_load_plugins(self):
        ray.init(ignore_reinit_error=True)
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
        remote_plugin = PluginActor.remote()
        remote_res = ray.get(remote_plugin.run.remote())
        self.assertEqual(remote_res[0], "Hello world")
        self.assertEqual(remote_res[1], "Hi")
        ray.shutdown(_exiting_interpreter=True)
