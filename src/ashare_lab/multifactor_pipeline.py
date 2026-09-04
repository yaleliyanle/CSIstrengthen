"""全A估计、中证1000应用的多因子回测流水线。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .charts import write_backtest_charts
from .execution import BacktestResult, ExecutionConfig, HistoricalAshareCosts, run_execution
from .metrics import compare_with_benchmark
from .multifactor import MultiFactorScores, build_multifactor_scores
from .pipeline import (
    _apply_empty_signal_policy,
    _benchmark_targets,
    _calendar,
    _executable_signals,
    _execution_setup,
    _members_for_signals,
    _normalize_weights,
    _resolve_factor_sources,
    make_catalogs,
)
from .portfolio import build_targets
from .schedule import make_schedule
from .settings import Config, FactorSpec
from .signals import apply_availability_lags, preprocess_factors


@dataclass(frozen=True)
class MultiFactorInputs:
    estimation: pl.DataFrame
    application: pl.DataFrame
    periods: pl.DataFrame
    specs: tuple[FactorSpec, ...]
    calendar: list[date]
    signal_dates: list[date]
    members: pl.DataFrame


@dataclass(frozen=True)
class MultiFactorArtifacts:
    method: str
    strategy: BacktestResult
    benchmark: BacktestResult
    comparison: pl.DataFrame
    summary: dict[str, Any]
    targets: pl.DataFrame
    scores: MultiFactorScores
    output_directory: Path


def load_multifactor_specs(config: Config) -> tuple[FactorSpec, ...]:
    """从冻结CSV读取因子；未配置文件时沿用 ``[[factors]]``。"""

    text = str(config.multifactor.get("factor_file", "")).strip()
    if not text:
        return config.factors
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"多因子清单不存在: {path}")
    table = pl.read_csv(path)
    if "selected" in table.columns:
        table = table.filter(pl.col("selected"))
    required = {"source"}
    if not required <= set(table.columns):
        raise ValueError("多因子清单至少需要 source 列")
    name_column = "raw_name" if "raw_name" in table.columns else "factor"
    if name_column not in table.columns:
        raise ValueError("多因子清单需要 raw_name 或 factor 列")
    direction_column = next(
        (
            column
            for column in (
                "suggested_direction_ic",
                "suggested_direction",
                "direction",
            )
            if column in table.columns
        ),
        None,
    )
    rows = []
    for row in table.iter_rows(named=True):
        raw_name = str(row[name_column])
        alias = str(row["factor"]) if "factor" in table.columns else raw_name
        rows.append(
            FactorSpec(
                source=str(row["source"]),
                name=raw_name,
                alias=alias if alias != raw_name else None,
                direction=float(row[direction_column]) if direction_column else 1.0,
            )
        )
    if not rows:
        raise ValueError("多因子清单为空")
    return tuple(rows)


def build_multifactor_inputs(config: Config) -> MultiFactorInputs:
    """在全A截面预处理并计算持有期标签，再取目标指数成分用于选股。"""

    catalog, factor_catalog = make_catalogs(config)
    specs = _resolve_factor_sources(load_multifactor_specs(config), factor_catalog)
    membership_lag = int(config.research.get("membership_lag", 0))
    full_calendar = _calendar(
        catalog,
        config.start - timedelta(days=membership_lag * 4 + 10),
        config.end + timedelta(days=100),
    )
    research_calendar = [day for day in full_calendar if config.start <= day <= config.end]
    schedule_calendar = (
        full_calendar
        if str(config.schedule.get("method", "monthly"))
        in {"weekly", "monthly", "quarterly"}
        else research_calendar
    )
    signal_dates = [
        day
        for day in make_schedule(schedule_calendar, config.schedule)
        if config.start <= day <= config.end
    ]
    if len(signal_dates) < 3:
        raise ValueError("多因子滚动研究至少需要三个调仓日")

    members = _members_for_signals(
        catalog, config.benchmark, signal_dates, full_calendar, membership_lag
    ).rename({"index_weight": "benchmark_weight"})
    members = _normalize_weights(members, "benchmark_weight")

    requests: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        assert spec.source is not None
        requests[spec.source].append(spec.name)
    max_lag = max(spec.availability_lag for spec in specs)
    factor_start = config.start - timedelta(days=max_lag * 4 + 10)

    # 全A交易日网格是 lag 的时间轴。DuckDB只把调仓日结果传回 Python。
    daily_grid = catalog.market(columns=[], start=factor_start, end=config.end)
    stocks = daily_grid.get_column("stock").unique().to_list()
    if hasattr(factor_catalog, "load_many_for_signals"):
        factors = factor_catalog.load_many_for_signals(
            requests=requests,
            lags={spec.name: spec.availability_lag for spec in specs},
            signal_dates=signal_dates,
            universe=daily_grid,
            start=factor_start,
            end=config.end,
        )
    else:
        factors = factor_catalog.load_many(
            requests,
            start=factor_start,
            end=config.end,
            stocks=stocks,
            universe=daily_grid,
        )
        factors = apply_availability_lags(factors, specs).filter(
            pl.col("date").is_in(signal_dates)
        )

    panel = factors
    if config.preprocess.get("neutralize_size", False):
        size = catalog.valuation(
            columns=["market_cap"],
            start=config.start,
            end=config.end,
            stocks=stocks,
        ).filter(pl.col("date").is_in(signal_dates))
        panel = panel.join(size, on=["date", "stock"], how="left")
    if config.preprocess.get("neutralize_industry", False):
        industry = catalog.industry(
            stocks=stocks, dates=signal_dates, drop_nulls=False, strict=False
        )
        panel = panel.join(industry, on=["date", "stock"], how="left")
    panel = preprocess_factors(panel, specs, config)
    factor_columns = [f"factor__{spec.output_name}" for spec in specs]
    processed = panel.select("date", "stock", *factor_columns).sort("date", "stock")

    market = catalog.market(
        columns=["adj_open", "adj_close"],
        start=config.start,
        end=full_calendar[-1],
        stocks=stocks,
    )
    labels, periods = period_forward_returns(
        market, signal_dates, full_calendar, str(config.execution["price"])
    )
    estimation = processed.join(labels, on=["date", "stock"], how="left")
    application = members.join(processed, on=["date", "stock"], how="left")
    return MultiFactorInputs(
        estimation=estimation,
        application=application,
        periods=periods,
        specs=specs,
        calendar=full_calendar,
        signal_dates=signal_dates,
        members=members,
    )


def period_forward_returns(
    market: pl.DataFrame,
    signal_dates: list[date],
    calendar: list[date],
    execution: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """计算本次调仓成交价到下次调仓成交价的复权收益。"""

    if execution not in {"same_close", "next_open"}:
        raise ValueError("execution 只能是 same_close 或 next_open")
    positions = {day: index for index, day in enumerate(calendar)}
    offset = 0 if execution == "same_close" else 1
    price_column = "adj_close" if execution == "same_close" else "adj_open"
    rows = []
    for current, following in zip(signal_dates, signal_dates[1:]):
        entry_index = positions[current] + offset
        exit_index = positions[following] + offset
        if exit_index >= len(calendar):
            continue
        rows.append(
            {
                "date": current,
                "entry_date": calendar[entry_index],
                "exit_date": calendar[exit_index],
            }
        )
    periods = pl.DataFrame(rows)
    prices = market.select(
        "date",
        "stock",
        pl.col(price_column).cast(pl.Float64, strict=False).alias("price"),
    ).filter((pl.col("price") > 0) & pl.col("price").is_finite())
    entries = periods.join(
        prices.rename({"date": "entry_date", "price": "entry_price"}),
        on="entry_date",
        how="inner",
    )
    labels = entries.join(
        prices.rename({"date": "exit_date", "price": "exit_price"}),
        on=["exit_date", "stock"],
        how="left",
    ).with_columns(
        (pl.col("exit_price") / pl.col("entry_price") - 1.0).alias(
            "forward_return"
        )
    ).select("date", "stock", "exit_date", "forward_return")
    return labels, periods


def run_multifactor_backtest(config: Config) -> MultiFactorArtifacts:
    """运行动态多因子合成，并复用原有组合、撮合、成本与基准。"""

    inputs = build_multifactor_inputs(config)
    scores = build_multifactor_scores(
        inputs.estimation, inputs.application, inputs.specs, config
    )
    scored_panel = inputs.application.join(
        scores.scores.select("date", "stock", "score"),
        on=["date", "stock"],
        how="inner",
    )
    start_text = str(config.multifactor.get("backtest_start", "")).strip()
    if start_text:
        scored_panel = scored_panel.filter(pl.col("date") >= date.fromisoformat(start_text))
    scored_dates = scored_panel.get_column("date").unique().sort().to_list()
    executable = _executable_signals(
        scored_dates, inputs.calendar, config.end, str(config.execution["price"])
    )
    if not executable:
        raise ValueError("没有通过暖机期且可以成交的多因子信号")
    scored_panel = scored_panel.filter(pl.col("date").is_in(executable))
    targets = _apply_empty_signal_policy(
        build_targets(scored_panel, config.portfolio),
        scored_panel,
        executable,
        str(config.portfolio.get("empty_signal", "error")),
    )

    catalog, _ = make_catalogs(config)
    backtest_calendar = [day for day in inputs.calendar if executable[0] <= day <= config.end]
    benchmark_targets = _benchmark_targets(
        catalog,
        config.benchmark,
        backtest_calendar,
        executable[0],
        str(config.execution["price"]),
        config.end,
        inputs.calendar,
    )
    stocks = sorted(
        set(inputs.members["stock"].to_list())
        | set(benchmark_targets["stock"].to_list())
    )
    market = catalog.market(
        columns=["open", "close", "adj_open", "adj_close", "trade_status"],
        start=backtest_calendar[0],
        end=backtest_calendar[-1],
        stocks=stocks,
    )
    limits = catalog.valuation(
        columns=["limit_status"],
        start=backtest_calendar[0],
        end=backtest_calendar[-1],
        stocks=stocks,
    )
    market = market.join(limits, on=["date", "stock"], how="left").rename(
        {"open": "raw_open", "close": "raw_close"}
    )
    execution, costs, trade_rule, lot_rounder = _execution_setup(config)
    strategy = run_execution(
        market,
        targets,
        config=execution,
        cost_model=costs,
        tradability_rule=trade_rule,
        lot_rounder=lot_rounder,
    )
    benchmark_execution = ExecutionConfig(
        profile="minimal",
        price=execution.price,
        initial_cash=execution.initial_cash,
        block_suspended=False,
        block_limit=False,
        t_plus_one=False,
        record_positions=False,
    )
    benchmark = run_execution(
        market,
        benchmark_targets,
        config=benchmark_execution,
        cost_model=HistoricalAshareCosts(0.0, 0.0, False, False),
    )
    comparison, summary = compare_with_benchmark(
        strategy.daily_nav,
        benchmark.daily_nav,
        annual_days=int(config.analytics["annual_days"]),
        risk_free_rate=float(config.analytics.get("risk_free_rate", 0.0)),
    )
    summary["run"] = {
        "method": scores.method,
        "factor_count": len(inputs.specs),
        "estimation_universe": "all_a",
        "application_universe": config.benchmark,
        "lookback_periods": int(config.multifactor["lookback_periods"]),
        "start": str(backtest_calendar[0]),
        "end": str(backtest_calendar[-1]),
        "signal_dates": len(executable),
        "execution": execution.price,
        "rebalance": config.schedule["method"],
        "fees": float(strategy.daily_nav["fees"].sum()),
        "benchmark_type": "synthetic_constituent_weight",
    }
    output = _write_multifactor(
        config,
        inputs,
        scores,
        scored_panel,
        strategy,
        benchmark,
        comparison,
        targets,
        summary,
    )
    return MultiFactorArtifacts(
        scores.method,
        strategy,
        benchmark,
        comparison,
        summary,
        targets,
        scores,
        output,
    )


def _write_multifactor(
    config: Config,
    inputs: MultiFactorInputs,
    scores: MultiFactorScores,
    panel: pl.DataFrame,
    strategy: BacktestResult,
    benchmark: BacktestResult,
    comparison: pl.DataFrame,
    targets: pl.DataFrame,
    summary: dict[str, Any],
) -> Path:
    output = Path(config.output["directory"]) / "multifactor" / scores.method
    output.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(output / "nav.csv")
    write_backtest_charts(comparison, output, config)
    strategy.daily_nav.write_csv(output / "strategy_daily.csv")
    benchmark.daily_nav.write_csv(output / "benchmark_daily.csv")
    strategy.trades.write_parquet(output / "trades.parquet")
    strategy.positions.write_parquet(output / "positions.parquet")
    strategy.rejections.write_parquet(output / "rejections.parquet")
    targets.write_parquet(output / "targets.parquet")
    panel.write_parquet(output / "signal_panel.parquet")
    inputs.periods.write_csv(output / "periods.csv")
    scores.ic_history.write_parquet(output / "factor_ic_history.parquet")
    scores.factor_weights.write_csv(output / "factor_weights.csv")
    scores.diagnostics.write_csv(output / "weight_diagnostics.csv")
    _yearly(comparison).write_csv(output / "yearly_performance.csv")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    resolved = config.effective_dict()
    resolved["factors"] = [asdict(spec) for spec in inputs.specs]
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return output


def _yearly(comparison: pl.DataFrame) -> pl.DataFrame:
    yearly = comparison.with_columns(
        pl.col("date").dt.year().alias("year")
    ).group_by("year").agg(
        ((1.0 + pl.col("strategy_return")).product() - 1.0).alias(
            "strategy_return"
        ),
        ((1.0 + pl.col("benchmark_return")).product() - 1.0).alias(
            "benchmark_return"
        ),
    )
    return yearly.with_columns(
        (
            (1.0 + pl.col("strategy_return"))
            / (1.0 + pl.col("benchmark_return"))
            - 1.0
        ).alias("excess_return")
    ).sort("year")


__all__ = [
    "MultiFactorArtifacts",
    "MultiFactorInputs",
    "build_multifactor_inputs",
    "load_multifactor_specs",
    "period_forward_returns",
    "run_multifactor_backtest",
]
