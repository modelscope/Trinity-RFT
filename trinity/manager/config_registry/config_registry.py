from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set

import streamlit as st

from trinity.utils.registry import Registry


class ConfigRegistry(Registry):
    """
    A registry for managing configuration settings and their associated functions.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._default_config = {}  # Stores default values for configs
        self._config_visibles = {}  # Stores visibles for config visibility
        self.unfinished_fields = set()

    def set_unfinished_fields(self, unfinished_fields: set):
        """
        Set the unfinished fields to track incomplete configurations.

        Args:
            unfinished_fields (set): Set of field names that are not yet configured.
        """
        self.unfinished_fields = unfinished_fields

    @property
    def default_config(self) -> dict:
        """
        Get the dictionary of default configuration values.
        """
        return self._default_config

    def get(self, config_name: str):
        """
        Retrieve a configuration function if its visible is met (if any).

        Args:
            config_name (str): Name of the configuration to retrieve.

        Returns:
            The configuration function if visibles are met, else None.
        """
        if config_name in self._config_visibles:
            if not self._config_visibles[config_name]():
                return None
        return super().get(config_name)

    def get_check_func(self, config_name: str):
        """
        Get the check function associated with a configuration.

        Args:
            config_name (str): Name of the configuration.

        Returns:
            The check function for the specified configuration.
        """
        check_func_name = f"check_{config_name}"
        return super().get(check_func_name)

    def get_configs(self, *config_names: str, columns_spec: List[int] = None):
        """
        Retrieve and display multiple configurations in Streamlit columns.

        Args:
            *config_names (str): Names of configurations to retrieve.
            columns_spec (List[int], optional): Configuration for Streamlit columns.
        """
        config_pair = []
        for config_name in config_names:
            config_func = self.get(config_name)
            if config_func is not None:
                config_pair.append((config_name, config_func))
        if len(config_pair) == 0:
            return

        if columns_spec is None:
            columns_spec = len(config_pair)
        columns = st.columns(columns_spec)
        for col, (_, config_func) in zip(columns, config_pair):
            with col:
                config_func()
        for config_name, _ in config_pair:
            check_func = self.get_check_func(config_name)
            if check_func is not None:
                check_func(unfinished_fields=self.unfinished_fields)

    def _register_config(
        self,
        config_name: str,
        config_func: Callable[[None], None],
        default_value: Optional[Any] = None,
        visible: Optional[Callable[[], bool]] = None,
        other_configs: Optional[Dict[str, Any]] = None,
    ):
        """
        Internal method to register a configuration and its associated function.

        Args:
            config_name (str): Name of the configuration.
            config_func (Callable): Function to set the configuration.
            default_value (Any, optional): Default value for the configuration.
            visible (Callable, optional): visible for when the config should be visible/applicable.
            other_configs (Dict[str, Any], optional): Additional configurations to register.
        """
        assert config_name not in self._default_config, f"{config_name} already exists."
        self._default_config[config_name] = default_value
        if visible is not None:
            self._config_visibles[config_name] = visible
        if other_configs is not None:
            for name, value in other_configs.items():
                assert name not in self._default_config, f"{name} already exists."
                self._default_config[name] = value
        super()._register_module(module_name=config_name, module_cls=config_func)

    def register_config(
        self,
        default_value: Optional[Any] = None,
        config_func: Optional[Callable[[None], None]] = None,
        visible: Optional[Callable[[], bool]] = None,
        other_configs: Optional[Dict[str, Any]] = None,
    ):
        """
        Decorator to register a configuration function.

        The function name must start with 'set_', and the part after 'set_' becomes the config name.

        Note: This function will automatically pass `key=config_name` as an argument to the
        registered configuration function. Ensure your function accepts this keyword argument.

        Args:
            default_value (Any, optional): Default value for the configuration.
            config_func (Callable, optional): The configuration function to register.
            visible (Callable, optional): visible for when the config should be visible.
            other_configs (Dict[str, Any], optional): Additional configurations to register.

        Returns:
            A decorator function if config_func is None, else the registered config function.
        """

        # if config_func is None, should return a decorator function
        def _register(config_func: Callable[[None], None]):
            config_name = config_func.__name__
            prefix = "set_"
            assert config_name.startswith(
                prefix
            ), f"Config function name should start with `{prefix}`, got {config_name}"
            config_name = config_name[len(prefix) :]
            config_func = partial(config_func, key=config_name)
            self._register_config(
                config_name=config_name,
                config_func=config_func,
                default_value=default_value,
                visible=visible,
                other_configs=other_configs,
            )
            return config_func

        if config_func is not None:
            return _register(config_func)
        return _register

    def _register_check(self, config_name: str, check_func: Callable[[Set, str], None]):
        """
        Internal method to register a check function for a configuration.

        Args:
            config_name (str): Name of the configuration to check.
            check_func (Callable): Function to check the configuration.
        """
        assert config_name in self._default_config, f"`{config_name}` is not registered."
        super()._register_module(module_name=f"check_{config_name}", module_cls=check_func)

    def register_check(self, check_func: Callable[[Set, str], None] = None):
        """
        Decorator to register a check function for a configuration.

        The function name must start with 'check_', and the part after 'check_' should match a config name.

        Note: This function will automatically pass `key=config_name` and `unfinished_fields=self.unfinished_fields` as an argument to the registered check function. Ensure your function accepts these keyword arguments.

        Args:
            check_func (Callable, optional): The check function to register.

        Returns:
            A decorator function if check_func is None, else the registered check function.
        """

        def _register(check_func: Callable[[Set, str], None]):
            config_name = check_func.__name__
            prefix = "check_"
            assert config_name.startswith(
                prefix
            ), f"Check function name must start with `{prefix}`, got {config_name}"
            config_name = config_name[len(prefix) :]
            check_func = partial(check_func, key=config_name)
            self._register_check(config_name, check_func)
            return check_func

        if check_func is not None:
            return _register(check_func)
        return _register


# Global registry for configuration generators
CONFIG_GENERATORS = ConfigRegistry("config_generators")
