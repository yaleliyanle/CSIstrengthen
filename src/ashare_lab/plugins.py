"""Minimal plugin entry point: configure ``module:function`` to use custom methods."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any


def import_callable(path: str) -> Callable[..., Any]:
    """Load ``package.module:function`` with clear error messages."""
    if ":" not in path:
        raise ValueError(f"Custom methods must use 'module:function'; received: {path!r}")
    module_name, function_name = path.split(":", 1)
    function = getattr(import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"{path!r} is not callable")
    return function


def resolve_method(
    name: str, builtins: Mapping[str, Callable[..., Any]]
) -> Callable[..., Any]:
    """Resolve a built-in short name first, then a ``module:function`` plugin."""
    if name in builtins:
        return builtins[name]
    return import_callable(name)
