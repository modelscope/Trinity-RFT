import os
import unittest


class TestSwanlabMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_env = {
            "SWANLAB_API_KEY": os.environ.get("SWANLAB_API_KEY"),
        }
        os.environ["SWANLAB_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxx"

    @classmethod
    def tearDownClass(cls):
        # Restore original environment variables
        for k, v in cls._original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @unittest.skip("Requires swanlab package and network access")
    def test_swanlab_monitor_smoke(self):
        from trinity.utils.monitor import SwanlabMonitor

        # Try creating the monitor; if swanlab isn't installed, __init__ will assert
        mon = SwanlabMonitor(
            project="trinity-smoke",
            group="cradle",
            name="swanlab-env",
            role="tester",
        )

        # Log a minimal metric to verify basic flow
        mon.log({"smoke/metric": 1.0}, step=1)
        mon.close()
