import unittest

import ray


class ImportUtils:
    def run(self):
        from trinity.common.workflows import WORKFLOWS, Workflow

        workflow_cls = WORKFLOWS.get("tests.utils.plugins.main.MainDummyWorkflow")
        assert issubclass(workflow_cls, Workflow)
        workflow = workflow_cls(task=None, model=None)
        res = workflow.run()
        assert res[0] == 0
        assert res[1] == "0"


class TestRegistry(unittest.TestCase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def test_dynamic_import(self):
        # test local import
        ImportUtils().run()
        # test remote import
        ray.get(ray.remote(ImportUtils).remote().run.remote())
