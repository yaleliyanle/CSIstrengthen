"""交易费用模型。

这里集中放置佣金、印花税和过户费。想改费用时，只需要阅读或替换本文件，
不必进入撮合循环。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from inspect import Parameter, signature
from functools import lru_cache
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

from .plugins import import_callable

Side: TypeAlias = Literal["buy", "sell"]
FeeRule: TypeAlias = Callable[[date, Side, float], float]


@dataclass(frozen=True)
class CostContext:
    """逐笔费用上下文；需要区分股票/交易所时用这一参数。"""

    date: date
    side: Side
    value: float
    stock: str


@dataclass(frozen=True)
class TradeCost:
    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0

    @property
    def total(self) -> float:
        return self.commission + self.stamp_tax + self.transfer_fee


class CostModel(Protocol):
    """自定义成本模型只需实现这个方法。"""

    def calculate(self, day: date, side: Side, value: float) -> TradeCost: ...


@dataclass(frozen=True)
class NoCosts:
    """``minimal`` profile 使用的零费用模型，也可单独注入。"""

    def calculate(self, _day: date, _side: Side, _value: float) -> TradeCost:
        return TradeCost()


def no_costs(_day: date, _side: Side, _value: float) -> TradeCost:
    """零费用函数版；配置中可写 ``ashare_lab.execution:no_costs``。"""

    return TradeCost()


@dataclass(frozen=True)
class HistoricalAshareCosts:
    """佣金、印花税和过户费。

    ``stamp_tax`` / ``transfer_fee`` 可设为 ``False`` 关闭，也可传入一个
    ``(date, side, value) -> fee`` 函数完全替换内置规则。
    """

    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax: bool | float | FeeRule = True
    transfer_fee: bool | float | FeeRule = True

    def __post_init__(self) -> None:
        if self.commission_rate < 0 or self.minimum_commission < 0:
            raise ValueError("佣金率和最低佣金不能为负数")

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any] | None = None
    ) -> "HistoricalAshareCosts":
        value = value or {}
        return cls(
            commission_rate=float(value.get("commission_rate", 0.0003)),
            minimum_commission=float(value.get("minimum_commission", 5.0)),
            stamp_tax=_fee_switch(value.get("stamp_tax", "historical"), "stamp_tax"),
            transfer_fee=_fee_switch(
                value.get("transfer_fee", "historical"), "transfer_fee"
            ),
        )

    def calculate(self, day: date, side: Side, value: float) -> TradeCost:
        if value <= 0:
            return TradeCost()
        commission = 0.0
        if self.commission_rate > 0:
            commission = max(self.minimum_commission, value * self.commission_rate)
        stamp = _component_fee(
            self.stamp_tax, historical_stamp_tax, day, side, value
        )
        transfer = _component_fee(
            self.transfer_fee, historical_transfer_fee, day, side, value
        )
        return TradeCost(commission, stamp, transfer)


def historical_stamp_tax_rate(day: date, side: Side) -> float:
    """返回历史印花税率；边界日按政策生效日处理。"""

    if day < date(2005, 1, 24):
        return 0.002
    if day < date(2007, 5, 30):
        return 0.001
    if day < date(2008, 4, 24):
        return 0.003
    if day < date(2008, 9, 19):
        return 0.001
    if side == "buy":
        return 0.0
    return 0.0005 if day >= date(2023, 8, 28) else 0.001


def historical_transfer_fee_rate(day: date, _side: Side) -> float:
    """当前数据所需的过户费历史：2022-04-29 起减半，双边收取。"""

    return 0.00001 if day >= date(2022, 4, 29) else 0.00002


def historical_stamp_tax(day: date, side: Side, value: float) -> float:
    return value * historical_stamp_tax_rate(day, side)


def historical_transfer_fee(day: date, side: Side, value: float) -> float:
    return value * historical_transfer_fee_rate(day, side)


def coerce_cost_model(
    value: CostModel
    | Callable[[date, Side, float], TradeCost]
    | Mapping[str, Any]
    | str
    | None,
) -> CostModel | Callable[[date, Side, float], TradeCost]:
    """把配置值变成统一的费用模型。"""

    if value is None:
        return HistoricalAshareCosts()
    if isinstance(value, Mapping):
        if value.get("method"):
            return coerce_cost_model(str(value["method"]))
        return HistoricalAshareCosts.from_dict(value)
    if isinstance(value, str):
        value = import_callable(value)
    if callable(value) or callable(getattr(value, "calculate", None)):
        return value
    raise TypeError("cost_model 应实现 calculate()、是可调用函数或 dict")


def trade_cost(
    model: CostModel | Callable[[date, Side, float], TradeCost],
    day: date,
    side: Side,
    value: float,
    stock: str = "",
) -> TradeCost:
    """调用费用模型；兼容旧三参数函数和单 ``CostContext`` 函数。"""

    if value <= 0:
        return TradeCost()
    target = (
        model.calculate  # type: ignore[union-attr]
        if callable(getattr(model, "calculate", None))
        else model
    )
    result = (
        target(CostContext(day, side, value, stock))
        if _takes_one_argument(target)
        else target(day, side, value)
    )
    if not isinstance(result, TradeCost):
        raise TypeError("cost_model 必须返回 TradeCost")
    amounts = (result.commission, result.stamp_tax, result.transfer_fee)
    if any(not np.isfinite(item) or item < 0 for item in amounts):
        raise ValueError("cost_model 返回的费用必须是非负有限数")
    return result


def _takes_one_argument(function: Callable[..., Any]) -> bool:
    try:
        return _takes_one_argument_cached(function)
    except TypeError:  # 少数不可哈希的 callable object
        return _inspect_one_argument(function)


@lru_cache(maxsize=128)
def _takes_one_argument_cached(function: Callable[..., Any]) -> bool:
    return _inspect_one_argument(function)


def _inspect_one_argument(function: Callable[..., Any]) -> bool:
    try:
        parameters = list(signature(function).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [
        item
        for item in parameters
        if item.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
        and item.default is Parameter.empty
    ]
    has_varargs = any(item.kind is Parameter.VAR_POSITIONAL for item in parameters)
    return len(positional) == 1 and not has_varargs


def _component_fee(
    setting: bool | float | FeeRule,
    builtin: FeeRule,
    day: date,
    side: Side,
    value: float,
) -> float:
    if setting is False:
        return 0.0
    if isinstance(setting, (int, float)) and not isinstance(setting, bool):
        rate = float(setting)
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("固定费率必须是非负有限数")
        return value * rate
    return float((builtin if setting is True else setting)(day, side, value))


def _fee_switch(value: Any, name: str) -> bool | FeeRule:
    if callable(value) or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        rate = float(value)
        if not np.isfinite(rate) or rate < 0:
            raise ValueError(f"{name} 固定费率必须是非负有限数")
        return lambda _day, _side, amount: amount * rate
    if isinstance(value, str) and ":" in value:
        return import_callable(value)  # type: ignore[return-value]
    if value is None or str(value).lower() in {"off", "false", "none", "disabled"}:
        return False
    if str(value).lower() in {"historical", "on", "true"}:
        return True
    raise ValueError(f"{name} 应为 historical/off 或费用函数")


__all__ = [
    "CostContext",
    "CostModel",
    "FeeRule",
    "HistoricalAshareCosts",
    "NoCosts",
    "Side",
    "TradeCost",
    "historical_stamp_tax",
    "historical_stamp_tax_rate",
    "historical_transfer_fee",
    "historical_transfer_fee_rate",
    "no_costs",
]
