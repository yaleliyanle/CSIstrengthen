"""Rebalance-date generation; add rules as functions with the same signature."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from .plugins import resolve_method


def daily(calendar: list[date], _: dict[str, Any]) -> list[date]:
    return calendar


def weekly(calendar: list[date], _: dict[str, Any]) -> list[date]:
    return _last_by(calendar, lambda day: day.isocalendar()[:2])


def monthly(calendar: list[date], _: dict[str, Any]) -> list[date]:
    return _last_by(calendar, lambda day: (day.year, day.month))


def quarterly(calendar: list[date], _: dict[str, Any]) -> list[date]:
    return _last_by(calendar, lambda day: (day.year, (day.month - 1) // 3))


def every_n_days(calendar: list[date], config: dict[str, Any]) -> list[date]:
    step = int(config.get("every_n_days", 5))
    if step < 1:
        raise ValueError("schedule.every_n_days must be positive")
    return calendar[::step]


BUILTINS = {
    "daily": daily,
    "weekly": weekly,
    "monthly": monthly,
    "quarterly": quarterly,
    "every_n_days": every_n_days,
}


def make_schedule(calendar: Iterable[date], config: dict[str, Any]) -> list[date]:
    days = sorted(set(calendar))
    method = resolve_method(str(config.get("method", "weekly")), BUILTINS)
    chosen = sorted(set(method(days, config)))
    unknown = set(chosen) - set(days)
    if unknown:
        raise ValueError(f"The schedule plugin returned a non-trading day: {min(unknown)}")
    return chosen


def _last_by(calendar: list[date], key: Any) -> list[date]:
    last: dict[Any, date] = {}
    for day in calendar:
        last[key(day)] = day
    return list(last.values())
