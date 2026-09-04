"""可直接复制修改的最小插件示例。

配置中的插件路径统一写成 ``ashare_lab.example_plugins:函数名``。
这些函数刻意只做一件事，方便在这里复制后改成自己的研究假设。
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
    """把当前可用的标准化因子等权平均，返回名为 ``score`` 的表达式。"""
    columns = [f"factor__{spec.output_name}" for spec in factor_specs]
    return pl.mean_horizontal(*(pl.col(column) for column in columns)).alias("score")


def top_twenty_percent(panel: pl.DataFrame, _config: dict[str, Any]) -> pl.DataFrame:
    """每个调仓日选择分数最高的 20%，并返回等权目标组合。"""
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
    """从传入的交易日历中选出每月第一个交易日。"""
    first: dict[tuple[int, int], date] = {}
    for day in sorted(calendar):
        first.setdefault((day.year, day.month), day)
    return list(first.values())


def block_new_stocks(context: OrderContext) -> str | None:
    """示例规则：交易状态为 ``N`` 时拒单，其余交给后续规则。"""
    return "custom_new_stock" if context.trade_status == "N" else None


def zero_cost(_day: date, _side: Side, _value: float) -> TradeCost:
    """示例成本模型：佣金、印花税、过户费都为零。"""
    return TradeCost()
