"""Load modules from custom directory"""

import importlib
import os
import shutil
import sys
from pathlib import Path

from trinity.utils.log import get_logger

logger = get_logger(__name__)


def load_plugins(plugin_dir: str) -> None:
    """
    Load plugin modules from a directory.
    """
    if plugin_dir is None:
        plugin_dir = Path(__file__).parent.parent / "plugins"
    if not os.path.exists(plugin_dir):
        logger.error(f"--plugin-dir [{plugin_dir}] does not exist.")
        return None
    if not os.path.isdir(plugin_dir):
        logger.error(f"--plugin-dir [{plugin_dir}] is not a directory.")
        return None

    logger.info(f"Loading plugin modules from [{plugin_dir}]...")
    for file in Path(plugin_dir).glob("*.py"):
        if file.name.startswith("__"):
            continue
        logger.info(f"Loading plugin modules from [{file}]...")
        # load modules from file
        load_from_file(os.path.join(plugin_dir, file))


def load_from_file(file_path: str):
    """
    Load modules from a Python file

    Args:
        file_path (`str`): The python file path.

    Returns:
        `Any`: The loaded module.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    full_module_name = f"trinity.plugins.{module_name}"

    spec = importlib.util.spec_from_file_location(full_module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)

    module.__package__ = "trinity.plugins"

    spec.loader.exec_module(module)

    if full_module_name in sys.modules:
        raise ImportError(f"Module {module_name} already exists.")
    sys.modules[full_module_name] = module
    shutil.copy2(file_path, Path(__file__).parent.parent / "plugins")
    logger.info(f"Load {file_path} as {full_module_name}")
    return module
