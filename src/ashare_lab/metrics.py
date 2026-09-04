"""净值与主动收益指标。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl


def performance_metrics(
    daily: pl.DataFrame,
    *,
    nav_column: str = "nav",
    return_column: str = "daily_return",
    annual_days: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | None]:
    """计算常见业绩指标；空样本返回明确错误。"""
    if daily.is_empty():
        raise ValueError("没有净值数据可计算指标")
    nav = daily[nav_column].cast(pl.Float64).to_numpy()
    returns = daily[return_column].cast(pl.Float64).fill_null(0.0).to_numpy()
    finite_returns = returns[np.isfinite(returns)]
    initial_nav = _initial_nav(daily, nav_column, return_column)
    total_return = float(nav[-1] / initial_nav - 1.0) if initial_nav else math.nan
    periods = max(len(nav), 1)
    annual_return = (
        float((nav[-1] / initial_nav) ** (annual_days / periods) - 1.0)
        if initial_nav > 0 and nav[-1] > 0
        else math.nan
    )
    volatility = float(np.std(finite_returns, ddof=1) * math.sqrt(annual_days)) if len(finite_returns) > 1 else 0.0
    excess = finite_returns - risk_free_rate / annual_days
    std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    sharpe = float(np.mean(excess) / std * math.sqrt(annual_days)) if std else None
    nav_path = np.concatenate(([initial_nav], nav))
    peak = np.maximum.accumulate(nav_path)
    drawdown = np.divide(
        nav_path, peak, out=np.ones_like(nav_path), where=peak != 0
    ) - 1.0
    max_drawdown = float(np.min(drawdown))
    return {
        "observations": len(nav),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown < 0 else None,
        "daily_win_rate": float(np.mean(finite_returns > 0)) if len(finite_returns) else None,
    }


def compare_with_benchmark(
    strategy: pl.DataFrame,
    benchmark: pl.DataFrame,
    annual_days: int = 252,
    risk_free_rate: float = 0.0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """按日对齐；``excess_nav`` 是策略净值 / 基准净值。"""
    strategy_base = _initial_nav(strategy, "nav", "daily_return")
    benchmark_base = _initial_nav(benchmark, "nav", "daily_return")
    strategy_summary = performance_metrics(
        strategy, annual_days=annual_days, risk_free_rate=risk_free_rate
    )
    benchmark_summary = performance_metrics(
        benchmark, annual_days=annual_days, risk_free_rate=risk_free_rate
    )
    combined = (
        strategy.select("date", pl.col("nav").alias("strategy_nav"), pl.col("daily_return").alias("strategy_return"))
        .join(
            benchmark.select("date", pl.col("nav").alias("benchmark_nav"), pl.col("daily_return").alias("benchmark_return")),
            on="date",
            how="inner",
        )
        .sort("date")
        .with_columns(
            (pl.col("strategy_nav") / strategy_base).alias("strategy_nav"),
            (pl.col("benchmark_nav") / benchmark_base).alias("benchmark_nav"),
            (pl.col("strategy_return") - pl.col("benchmark_return")).alias("active_return"),
        )
        .with_columns(
            (pl.col("strategy_nav") / pl.col("benchmark_nav")).alias("excess_nav")
        )
    )
    active = combined["active_return"].fill_null(0.0).to_numpy()
    active_std = float(np.std(active, ddof=1)) if len(active) > 1 else 0.0
    information_ratio = float(np.mean(active) / active_std * math.sqrt(annual_days)) if active_std else None
    summary = {
        "strategy": strategy_summary,
        "benchmark": benchmark_summary,
        "active": {
            "total_excess_return": float(combined["excess_nav"][-1] - 1.0),
            "information_ratio": information_ratio,
            "tracking_error": active_std * math.sqrt(annual_days),
        },
    }
    return combined, summary


def _initial_nav(daily: pl.DataFrame, nav_column: str, return_column: str) -> float:
    """优先读撮合器写出的本金；否则由首日收益反推。"""
    if "initial_nav" in daily.columns:
        values = daily["initial_nav"].drop_nulls()
        if len(values):
            return float(values[0])
    first_nav = float(daily[nav_column][0])
    first_return = daily[return_column][0]
    if first_return is not None:
        value = float(first_return)
        if math.isfinite(value) and value > -1.0:
            return first_nav / (1.0 + value)
    return first_nav
