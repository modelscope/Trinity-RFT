import sys
import unittest
from unittest import mock

from tests.tools import get_template_config
from trinity.cli import launcher
from trinity.common.constants import (
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    LOG_NODE_IP_ENV_VAR,
    PLUGIN_DIRS_ENV_VAR,
)


class TestLauncherMain(unittest.TestCase):
    def setUp(self):
        self._orig_argv = sys.argv.copy()

    def tearDown(self):
        sys.argv = self._orig_argv

    @mock.patch("trinity.cli.launcher.explore")
    @mock.patch("trinity.cli.launcher.train")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.bench")
    @mock.patch("trinity.cli.launcher.load_config")
    def test_main_run_command(self, mock_load, mock_bench, mock_both, mock_train, mock_explore):
        config = get_template_config()
        mapping = {
            "explore": mock_explore,
            "train": mock_train,
            "both": mock_both,
            "bench": mock_bench,
        }
        for mode in ["explore", "train", "both", "bench"]:
            config.mode = mode
            mock_load.return_value = config
            with mock.patch(
                "argparse.ArgumentParser.parse_args",
                return_value=mock.Mock(
                    command="run", config="dummy.yaml", dlc=False, plugin_dir=None
                ),
            ):
                launcher.main()
            mock_load.assert_called_once_with("dummy.yaml")
            mapping[mode].assert_called_once_with(config)
            mock_load.reset_mock()
            mapping[mode].reset_mock()

    @mock.patch("trinity.cli.launcher.setup_ray_cluster")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.load_config")
    def test_main_run_in_dlc(self, mock_load, mock_both, mock_setup):
        config = get_template_config()
        config.mode = "both"
        config.log.level = "WARNING"
        config.log.group_by_node = True
        mock_load.return_value = config
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=mock.Mock(
                command="run", config="dummy.yaml", dlc=True, plugin_dir="/path/to/plugins"
            ),
        ):
            launcher.main()
        mock_load.assert_called_once_with("dummy.yaml")
        mock_both.assert_called_once_with(config)
        mock_setup.assert_called_once_with(
            namespace=config.ray_namespace,
            envs={
                PLUGIN_DIRS_ENV_VAR: "/path/to/plugins",
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "WARNING",
                LOG_NODE_IP_ENV_VAR: "1",
            },
        )

    @mock.patch("trinity.cli.launcher.studio")
    def test_main_studio_command(self, mock_studio):
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=mock.Mock(command="studio", port=9999),
        ):
            launcher.main()
        mock_studio.assert_called_once_with(9999)


if __name__ == "__main__":
    unittest.main()
