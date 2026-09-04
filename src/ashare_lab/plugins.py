"""很小的插件入口：配置写 ``模块:函数`` 即可使用自定义方法。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any


def import_callable(path: str) -> Callable[..., Any]:
    """载入 ``package.module:function``，错误信息保持直白。"""
    if ":" not in path:
        raise ValueError(f"自定义方法应写成 '模块:函数'，收到: {path!r}")
    module_name, function_name = path.split(":", 1)
    function = getattr(import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"{path!r} 不是可调用函数")
    return function


def resolve_method(
    name: str, builtins: Mapping[str, Callable[..., Any]]
) -> Callable[..., Any]:
    """先查内置短名称，再查 ``模块:函数`` 插件。"""
    if name in builtins:
        return builtins[name]
    return import_callable(name)
