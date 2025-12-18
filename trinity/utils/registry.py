import traceback
from typing import Any, Type

from trinity.utils.log import get_logger


class Registry(object):
    """A class for registry."""

    def __init__(self, name: str, default_mapping: dict = {}):
        """
        Args:
            name (`str`): The name of the registry.
            default_mapping (`dict`): Default mapping from module names to module paths (strings).
        """
        self._name = name
        self._modules = {}
        self._default_mapping = default_mapping
        self.logger = get_logger()

    @property
    def name(self) -> str:
        """
        Get name of current registry.

        Returns:
            `str`: The name of current registry.
        """
        return self._name

    @property
    def modules(self) -> dict:
        """
        Get all modules in current registry.

        Returns:
            `dict`: A dict storing modules in current registry.
        """
        return self._modules

    def get(self, module_key) -> Any:
        """
        Get module named module_key from in current registry. If not found,
        return None.

        Args:
            module_key (`str`): specified module name

        Returns:
            `Any`: the module object
        """
        module = self._modules.get(module_key, None)
        if module is None:
            # try to get from default mapping
            if module_key in self._default_mapping:
                module_path, class_name = self._default_mapping[module_key].rsplit(".", 1)
                try:
                    module = self._dynamic_import(module_path, class_name)
                except Exception:
                    self.logger.error(
                        f"Failed to dynamically import {class_name} from {module_path}:\n"
                        + traceback.format_exc()
                    )
                    raise ImportError(f"Cannot dynamically import {class_name} from {module_path}")
            # try to get from string path
            elif isinstance(module_key, str) and "." in module_key:
                module_path, class_name = module_key.rsplit(".", 1)
                try:
                    module = self._dynamic_import(module_path, class_name)
                except Exception:
                    self.logger.error(
                        f"Failed to dynamically import {class_name} from {module_path}:\n"
                        + traceback.format_exc()
                    )
                    raise ImportError(f"Cannot dynamically import {class_name} from {module_path}")
                self._register_module(module_name=module_key, module_cls=module)
            elif module_key is None:
                self.logger.info("Empty module key, return None")
                return None
            else:
                raise ValueError(f"Invalid module key: {module_key}")
        return module

    def _register_module(self, module_name=None, module_cls=None, force=False):
        """
        Register module to registry.
        """

        if module_name is None:
            module_name = module_cls.__name__

        if module_name in self._modules and not force:
            self.logger.warning(
                f"{module_name} is already registered in {self._name}, "
                f"if you want to override it, please set force=True."
            )
            raise KeyError(f"{module_name} is already registered in {self._name}")

        self._modules[module_name] = module_cls
        module_cls._name = module_name

    def register_module(self, module_name: str, module_cls: Type = None, force=False, lazy=False):
        """
        Register module class object to registry with the specified module name.

        Args:
            module_name (`str`): The module name.
            module_cls (`Type`): module class object
            force (`bool`): Whether to override an existing class with
                    the same name. Default: False.
            lazy (`bool`): Whether to register the module class object lazily.
                    Default: False.

        Example:

            .. code-block:: python

                WORKFLOWS = Registry("workflows")

                # register a module using decorator
                @WORKFLOWS.register_module(name="workflow_name")
                class MyWorkflow(Workflow):
                    pass

                # or register a module directly
                WORKFLOWS.register_module(
                    name="workflow_name",
                    module_cls=MyWorkflow,
                    force=True,
                )
        """
        if not (module_name is None or isinstance(module_name, str)):
            raise TypeError(f"module_name must be either of None, str," f"got {type(module_name)}")
        if module_cls is not None:
            self._register_module(module_name=module_name, module_cls=module_cls, force=force)
            return module_cls

        # if module_cls is None, should return a decorator function
        def _register(module_cls):
            """
            Register module class object to registry.

            Args:
                module_cls (`Type`): module class object
            Returns:
                `Type`: Decorated module class object.
            """
            self._register_module(module_name=module_name, module_cls=module_cls, force=force)
            return module_cls

        return _register

    def _dynamic_import(self, module_path: str, class_name: str) -> Type:
        """
        Dynamically import a module class object from the specified module path.

        Args:
            module_path (`str`): The module path. For example, "my_package.my_module".
            class_name (`str`): The class name. For example, "MyWorkflow".

        Returns:
            `Type`: The imported module class object.
        """
        import importlib

        module = importlib.import_module(module_path)
        module_cls = getattr(module, class_name)
        return module_cls
