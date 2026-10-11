"""Transaction cost models.

Commission, stamp duty, and transfer fees are defined here. Modify or replace this file
to change costs without touching the execution loop.
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
    """Per-trade cost context for models that distinguish stocks or exchanges."""

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
    """Custom cost models only need to implement this method."""

    def calculate(self, day: date, side: Side, value: float) -> TradeCost: ...


@dataclass(frozen=True)
class NoCosts:
    """Zero-cost model used by the ``minimal`` profile; it can also be injected independently."""

    def calculate(self, _day: date, _side: Side, _value: float) -> TradeCost:
        return TradeCost()


def no_costs(_day: date, _side: Side, _value: float) -> TradeCost:
    """Functional zero-cost model; configure it as ``ashare_lab.execution:no_costs``."""

    return TradeCost()


@dataclass(frozen=True)
class HistoricalAshareCosts:
    """Commission, stamp duty, and transfer fees.

    Set ``stamp_tax`` / ``transfer_fee`` to ``False`` to disable them, or supply a
    ``(date, side, value) -> fee`` function to replace the built-in rules.
    """

    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax: bool | float | FeeRule = True
    transfer_fee: bool | float | FeeRule = True

    def __post_init__(self) -> None:
        if self.commission_rate < 0 or self.minimum_commission < 0:
            raise ValueError("Commission rate and minimum commission must be nonnegative")

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
    """Return historical stamp-duty rates, using policy effective dates as boundaries."""

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
    """Transfer-fee history used by this dataset: halved on 2022-04-29 and charged on both sides."""

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
    """Convert configuration values into a consistent cost model."""

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
    raise TypeError("cost_model must implement calculate(), be callable, or be a dict")


def trade_cost(
    model: CostModel | Callable[[date, Side, float], TradeCost],
    day: date,
    side: Side,
    value: float,
    stock: str = "",
) -> TradeCost:
    """Call cost models supporting either the legacy three arguments or a single ``CostContext``."""

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
        raise TypeError("cost_model must return TradeCost")
    amounts = (result.commission, result.stamp_tax, result.transfer_fee)
    if any(not np.isfinite(item) or item < 0 for item in amounts):
        raise ValueError("cost_model must return nonnegative finite fees")
    return result


def _takes_one_argument(function: Callable[..., Any]) -> bool:
    try:
        return _takes_one_argument_cached(function)
    except TypeError:  # Some callable objects are unhashable
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
            raise ValueError("Fixed fee rates must be nonnegative and finite")
        return value * rate
    return float((builtin if setting is True else setting)(day, side, value))


def _fee_switch(value: Any, name: str) -> bool | FeeRule:
    if callable(value) or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        rate = float(value)
        if not np.isfinite(rate) or rate < 0:
            raise ValueError(f"{name} Fixed fee rates must be nonnegative and finite")
        return lambda _day, _side, amount: amount * rate
    if isinstance(value, str) and ":" in value:
        return import_callable(value)  # type: ignore[return-value]
    if value is None or str(value).lower() in {"off", "false", "none", "disabled"}:
        return False
    if str(value).lower() in {"historical", "on", "true"}:
        return True
    raise ValueError(f"{name} must be historical/off or a fee function")


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
