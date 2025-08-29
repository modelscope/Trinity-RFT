from typing import Any, Type


# TODO: support lazy load
# e.g. @MODULES.register_module("name", lazy=True)
class Registry(object):
    """A class for registry."""

    def __init__(self, name: str):
        """
        Args:
            name (`str`): The name of the registry.
        """
        self._name = name
        self._modules = {}

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
        return self._modules.get(module_key, None)

    def _register_module(self, module_name=None, module_cls=None, force=False):
        """
        Register module to registry.
        """

        if module_name is None:
            module_name = module_cls.__name__

        if module_name in self._modules and not force:
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
