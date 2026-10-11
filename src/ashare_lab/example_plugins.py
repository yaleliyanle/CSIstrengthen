"""Minimal plugin examples ready to copy and customize.

Configure plugin paths as ``ashare_lab.example_plugins:function_name``.
Each function does one thing so it is easy to adapt to your research assumptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import polars as pl

from .execution import OrderContext, Side, TradeCost


def average_available_factors(
    panel: pl.DataFrame, factor_specs: Sequence[Any], _config: Any
) -> pl.Expr:
    """Average available standardized factors equally and return an expression named ``score``."""
    columns = [f"factor__{spec.output_name}" for spec in factor_specs]
    return pl.mean_horizontal(*(pl.col(column) for column in columns)).alias("score")


def top_twenty_percent(panel: pl.DataFrame, _config: dict[str, Any]) -> pl.DataFrame:
    """Select the highest-scoring 20% on each rebalance date and return equal target weights."""
    selected = (
        panel.filter(pl.col("score").is_finite())
        .with_columns(
            pl.col("score")
            .rank("ordinal", descending=True)
            .over("date")
            .alias("_rank"),
            pl.len().over("date").alias("_count"),
        )
        .filter(pl.col("_rank") <= (pl.col("_count") * 0.2).ceil().clip(lower_bound=1))
    )
    return selected.select(
        "date",
        "stock",
        (1.0 / pl.len().over("date")).alias("target_weight"),
    )


def first_trading_day_each_month(
    calendar: list[date], _config: dict[str, Any]
) -> list[date]:
    """Select the first trading day of each month from the supplied calendar."""
    first: dict[tuple[int, int], date] = {}
    for day in sorted(calendar):
        first.setdefault((day.year, day.month), day)
    return list(first.values())


def block_new_stocks(context: OrderContext) -> str | None:
    """Example rule: reject orders with trade status ``N``; defer other cases to subsequent rules."""
    return "custom_new_stock" if context.trade_status == "N" else None


def zero_cost(_day: date, _side: Side, _value: float) -> TradeCost:
    """Example cost model with zero commission, stamp duty, and transfer fees."""
    return TradeCost()
