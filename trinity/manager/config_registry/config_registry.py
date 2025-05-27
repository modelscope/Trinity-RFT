from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set

import streamlit as st

from trinity.utils.registry import Registry


class ConfigRegistry(Registry):
    def __init__(self, name: str):
        super().__init__(name)
        self._default_config = {}
        self._config_conditions = {}

    def set_unfinished_fields(self, unfinished_fields: set):
        self.unfinished_fields = unfinished_fields

    @property
    def default_config(self) -> dict:
        return self._default_config

    def get(self, config_name: str):
        if config_name in self._config_conditions:
            if not self._config_conditions[config_name]():
                return None
        return super().get(config_name)

    def get_check_func(self, config_name: str):
        check_func_name = f"check_{config_name}"
        return super().get(check_func_name)

    def get_configs(self, *config_names: str, columns_config: List[int] = None):
        config_pair = []
        for config_name in config_names:
            config_func = self.get(config_name)
            if config_func is not None:
                config_pair.append((config_name, config_func))
        if len(config_pair) == 0:
            return

        if columns_config is None:
            columns_config = len(config_pair)
        columns = st.columns(columns_config)
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
        condition: Optional[Callable[[], bool]] = None,
        other_configs: Optional[Dict[str, Any]] = None,
    ):
        assert config_name not in self._default_config, f"{config_name} already exists."
        self._default_config[config_name] = default_value
        if condition is not None:
            self._config_conditions[config_name] = condition
        if other_configs is not None:
            for name, value in other_configs.items():
                assert name not in self._default_config, f"{name} already exists."
                self._default_config[name] = value
        super()._register_module(module_name=config_name, module_cls=config_func)

    def register_config(
        self,
        default_value: Optional[Any] = None,
        config_func: Optional[Callable[[None], None]] = None,
        condition: Optional[Callable[[], bool]] = None,
        other_configs: Optional[Dict[str, Any]] = None,
    ):
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
                condition=condition,
                other_configs=other_configs,
            )
            return config_func

        if config_func is not None:
            return _register(config_func)
        return _register

    def _register_check(self, config_name: str, check_func: Callable[[Set, str], None]):
        assert config_name in self._default_config, f"`{config_name}` is not registered."
        super()._register_module(module_name=f"check_{config_name}", module_cls=check_func)

    def register_check(self, check_func: Callable[[Set, str], None] = None):
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


CONFIG_GENERATORS = ConfigRegistry("config_generators")
