"""把数据、信号、组合和撮合接成两条清晰流水线。"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .analytics import FactorReport, analyze_factor, forward_returns
from .charts import write_backtest_charts
from .data import DataCatalog, FactorCatalog
from .duckdb_data import DuckDBDataCatalog, DuckDBFactorCatalog
from .execution import (
    BacktestResult,
    ExecutionConfig,
    HistoricalAshareCosts,
    run_execution,
)
from .metrics import compare_with_benchmark
from .plugins import import_callable
from .portfolio import build_targets
from .schedule import make_schedule
from .selection import SelectionResult, select_factors
from .settings import Config, FactorSpec
from .signals import apply_availability_lags, combine_scores, preprocess_factors
from .validation import ValidationResult, dataset_split, validate_two_stage


@dataclass(frozen=True)
class ResearchPanel:
    data: pl.DataFrame
    specs: tuple[FactorSpec, ...]
    calendar: list[date]
    signal_dates: list[date]
    members: pl.DataFrame


@dataclass(frozen=True)
class BacktestArtifacts:
    strategy: BacktestResult
    benchmark: BacktestResult
    comparison: pl.DataFrame
    summary: dict[str, Any]
    targets: pl.DataFrame
    output_directory: Path


def make_catalogs(config: Config) -> tuple[Any, Any]:
    """路径只在这里组装，数据层和策略层不认识彼此。"""
    data = config.data
    market = Path(data["market"])
    common = {
        "root": Path("."),
        "market_name": str(market),
        "valuation_name": str(Path(data["valuation"])),
        "index_name": str(Path(data["index_members"])),
        "industry_name": str(Path(data["industry"])),
    }
    if data.get("engine", "polars") == "duckdb":
        threads = int(data.get("duckdb_threads", 4))
        catalog = DuckDBDataCatalog(**common, threads=threads)
        factors = DuckDBFactorCatalog(
            data["factor_root"],
            batch_size=int(data.get("batch_size", 16)),
            threads=threads,
        )
        return catalog, factors
    catalog = DataCatalog(
        **common,
    )
    return catalog, FactorCatalog(data["factor_root"])


def build_research_panel(config: Config) -> ResearchPanel:
    """构建只含 point-in-time 指数成分和调仓日的因子截面。"""
    catalog, factor_catalog = make_catalogs(config)
    longest_horizon = max(
        max(config.analytics.get("horizons", [1])),
        int(config.selection.get("horizon", 1)),
    )
    buffer_days = max(100, longest_horizon * 4 + 10)
    membership_lag = int(config.research.get("membership_lag", 0))
    full_calendar = _calendar(
        catalog,
        config.start - timedelta(days=membership_lag * 4 + 10),
        config.end + timedelta(days=buffer_days),
    )
    research_calendar = [
        day for day in full_calendar if config.start <= day <= config.end
    ]
    if not research_calendar:
        raise ValueError("请求区间内没有行情交易日")
    # 周/月/季频必须先看见下一周期，才能确认区间末日是否真的是期末交易日。
    # daily/every_n_days/自定义插件仍从 research.start 锚定，行为更直观。
    schedule_method = str(config.schedule.get("method", "weekly"))
    schedule_calendar = (
        full_calendar
        if schedule_method in {"weekly", "monthly", "quarterly"}
        else research_calendar
    )
    signal_dates = [
        day
        for day in make_schedule(schedule_calendar, config.schedule)
        if config.start <= day <= config.end
    ]
    if not signal_dates:
        raise ValueError("调仓规则没有生成任何日期")

    members = _members_for_signals(
        catalog,
        config.benchmark,
        signal_dates,
        full_calendar,
        membership_lag,
    ).rename({"index_weight": "benchmark_weight"})
    members = _normalize_weights(members, "benchmark_weight")
    stocks = members["stock"].unique().to_list()
    specs = _resolve_factor_sources(config.factors, factor_catalog)
    requests: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        assert spec.source is not None
        requests[spec.source].append(spec.name)

    max_lag = max(spec.availability_lag for spec in specs)
    factor_start = config.start - timedelta(days=max_lag * 4 + 10)
    # 以行情交易日键为左表：即使只选深度学习等稀疏因子，lag=1 仍表示
    # 一个真实交易日，而不是“下一条碰巧存在的因子记录”。
    daily_grid = catalog.market(
        columns=[], start=factor_start, end=config.end, stocks=stocks
    )
    if hasattr(factor_catalog, "load_many_for_signals"):
        daily_factors = factor_catalog.load_many_for_signals(
            requests=requests,
            lags={spec.name: spec.availability_lag for spec in specs},
            signal_dates=signal_dates,
            universe=daily_grid,
            start=factor_start,
            end=config.end,
        )
    else:
        daily_factors = factor_catalog.load_many(
            requests,
            start=factor_start,
            end=config.end,
            stocks=stocks,
            universe=daily_grid,
        )
        daily_factors = apply_availability_lags(daily_factors, specs).filter(
            pl.col("date").is_in(signal_dates)
        )
    panel = members.join(daily_factors, on=["date", "stock"], how="left")

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
    panel = combine_scores(panel, specs, config)
    return ResearchPanel(
        data=panel.sort(["date", "stock"]),
        specs=specs,
        calendar=full_calendar,
        signal_dates=signal_dates,
        members=members,
    )


def run_factor_research(config: Config) -> list[FactorReport]:
    """对配置中的每个因子、每个持有期执行单因子检测并落盘。"""
    research = build_research_panel(config)
    catalog, _ = make_catalogs(config)
    horizons = [int(item) for item in config.analytics["horizons"]]
    end = research.calendar[-1]
    stocks = research.members["stock"].unique().to_list()
    market = catalog.market(
        columns=["adj_open", "adj_close"],
        start=config.start,
        end=end,
        stocks=stocks,
    )
    labels = forward_returns(
        market,
        horizons,
        str(config.execution["price"]),
    )
    panel = research.data.join(labels, on=["date", "stock"], how="left")
    reports = []
    for spec in research.specs:
        factor_column = f"factor__{spec.output_name}"
        for horizon in horizons:
            reports.append(
                analyze_factor(
                    panel,
                    factor_column,
                    f"forward_{horizon}d",
                    config,
                    calendar=research.calendar,
                )
            )
    _write_factor_reports(reports, config, research.specs)
    return reports


def run_factor_selection(config: Config) -> SelectionResult:
    """在指定研究区间筛选因子，并保存一张易读结果表。"""
    research = build_research_panel(config)
    catalog, _ = make_catalogs(config)
    horizon = int(config.selection.get("horizon", 20))
    stocks = research.members["stock"].unique().to_list()
    market = catalog.market(
        columns=["adj_open", "adj_close"],
        start=config.start,
        end=research.calendar[-1],
        stocks=stocks,
    )
    labels = forward_returns(
        market, [horizon], str(config.execution["price"])
    )
    panel = research.data.join(labels, on=["date", "stock"], how="left")
    columns = [f"factor__{spec.output_name}" for spec in research.specs]
    result = select_factors(panel, columns, f"forward_{horizon}d", config)
    _write_selection(result, config, research.specs)
    return result


def run_factor_validation(config: Config) -> ValidationResult:
    """运行 IC→Lasso 两层验证；不会读取 test/live 的因子或收益。"""
    train, validation, _, _ = dataset_split(config)
    raw = config.to_dict()
    raw["research"]["start"] = train.start.isoformat()
    raw["research"]["end"] = validation.end.isoformat()
    development_config = Config.from_dict(raw)

    research = build_research_panel(development_config)
    catalog, _ = make_catalogs(development_config)
    horizon = int(config.selection.get("horizon", 20))
    stocks = research.members["stock"].unique().to_list()
    market = catalog.market(
        columns=["adj_open", "adj_close"],
        start=train.start,
        end=validation.end,
        stocks=stocks,
    )
    labels = forward_returns(
        market, [horizon], str(config.execution["price"])
    )
    panel = research.data.join(labels, on=["date", "stock"], how="left")
    columns = [f"factor__{spec.output_name}" for spec in research.specs]
    calendar = [
        day
        for day in research.calendar
        if train.start <= day <= validation.end
    ]
    result = validate_two_stage(
        panel,
        columns,
        f"forward_{horizon}d",
        calendar,
        config,
    )
    _write_validation(result, config, research.specs)
    return result


def run_backtest(config: Config) -> BacktestArtifacts:
    """运行指数增强组合和无摩擦基准，并输出完整审计文件。"""
    research = build_research_panel(config)
    if config.execution["price"] == "same_close" and not config.execution.get(
        "allow_same_close", False
    ):
        warnings.warn(
            "same_close 假设因子在当日收盘成交前可得；请确认没有未来函数。",
            stacklevel=2,
        )

    backtest_calendar = [
        day for day in research.calendar if config.start <= day <= config.end
    ]
    if not backtest_calendar:
        raise ValueError("请求区间内没有可回测交易日")
    executable_signals = _executable_signals(
        research.signal_dates,
        research.calendar,
        config.end,
        str(config.execution["price"]),
    )
    if not executable_signals:
        raise ValueError("区间内没有能完成成交的调仓日")
    signal_panel = research.data.filter(pl.col("date").is_in(executable_signals))
    targets = _apply_empty_signal_policy(
        build_targets(signal_panel, config.portfolio),
        signal_panel,
        executable_signals,
        str(config.portfolio.get("empty_signal", "error")),
    )
    catalog, _ = make_catalogs(config)
    benchmark_targets = _benchmark_targets(
        catalog,
        config.benchmark,
        backtest_calendar,
        executable_signals[0],
        str(config.execution["price"]),
        config.end,
        research.calendar,
    )
    stocks = sorted(
        set(research.members["stock"].to_list())
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
        use_eod_limit_for_open=False,
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
        "start": str(backtest_calendar[0]),
        "end": str(backtest_calendar[-1]),
        "signal_dates": len(executable_signals),
        "stocks": len(stocks),
        "execution": execution.price,
        "rebalance": config.schedule["method"],
        "benchmark_code": config.benchmark,
        "benchmark_type": "synthetic_total_return_proxy",
        "rules_profile": config.rules.get("profile", "standard"),
        "trades": strategy.trades.height,
        "rejections": strategy.rejections.height,
        "fees": float(strategy.daily_nav["fees"].sum()),
    }
    output = _write_backtest(
        config,
        strategy,
        benchmark,
        comparison,
        targets,
        signal_panel,
        summary,
        research.specs,
    )
    return BacktestArtifacts(
        strategy=strategy,
        benchmark=benchmark,
        comparison=comparison,
        summary=summary,
        targets=targets,
        output_directory=output,
    )


def _calendar(catalog: DataCatalog, start: date, end: date) -> list[date]:
    return (
        catalog.market(columns=[], start=start, end=end)
        .get_column("date")
        .unique()
        .sort()
        .to_list()
    )


def _resolve_factor_sources(
    specs: tuple[FactorSpec, ...], catalog: FactorCatalog
) -> tuple[FactorSpec, ...]:
    listing = catalog.list_factors()
    result = []
    for spec in specs:
        if spec.source:
            if spec.name not in catalog.factor_names(spec.source):
                raise KeyError(f"{spec.source} 没有因子 {spec.name!r}")
            result.append(spec)
            continue
        sources = listing.filter(pl.col("name") == spec.name)["source"].to_list()
        if len(sources) != 1:
            raise ValueError(
                f"因子 {spec.name!r} 匹配到 {sources}；请在配置填写 source"
            )
        result.append(replace(spec, source=sources[0]))
    return tuple(result)


def _normalize_weights(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    total = pl.col(column).sum().over("date")
    return frame.filter(pl.col(column) >= 0).with_columns(
        pl.when(total > 0).then(pl.col(column) / total).otherwise(0.0).alias(column)
    )


def _members_for_signals(
    catalog: DataCatalog,
    benchmark: str,
    signals: list[date],
    calendar: list[date],
    lag: int,
) -> pl.DataFrame:
    positions = {day: index for index, day in enumerate(calendar)}
    if any(positions[day] < lag for day in signals):
        raise ValueError("交易日历不足以应用 research.membership_lag")
    lookup_dates = [calendar[positions[day] - lag] for day in signals]
    mapping = pl.DataFrame({"date": lookup_dates, "signal_date": signals})
    return (
        catalog.index_members(lookup_dates, index=benchmark)
        .join(mapping, on="date", how="left")
        .drop("date")
        .rename({"signal_date": "date"})
        .sort(["date", "stock"])
    )


def _benchmark_targets(
    catalog: DataCatalog,
    benchmark: str,
    backtest_calendar: list[date],
    first_signal: date,
    price: str,
    end: date,
    extended_calendar: list[date],
) -> pl.DataFrame:
    """指数只在成分快照变化时复位权重，期间让权重随价格自然漂移。"""
    dates = [day for day in backtest_calendar if day >= first_signal]
    expanded = catalog.index_members(dates, index=benchmark).rename(
        {"index_weight": "benchmark_weight"}
    )
    snapshots = (
        expanded.select("date", "snapshot_date")
        .unique()
        .sort("date")
        .with_columns(pl.col("snapshot_date").shift(1).alias("_previous"))
    )
    rebalance_dates = snapshots.filter(
        pl.col("_previous").is_null()
        | (pl.col("snapshot_date") != pl.col("_previous"))
    )["date"].to_list()
    rebalance_dates = _executable_signals(
        rebalance_dates, extended_calendar, end, price
    )
    return _normalize_weights(
        expanded.filter(pl.col("date").is_in(rebalance_dates)),
        "benchmark_weight",
    ).select(
        "date", "stock", pl.col("benchmark_weight").alias("target_weight")
    )


def _executable_signals(
    signals: list[date],
    calendar: list[date],
    end: date,
    price: str,
) -> list[date]:
    """丢弃在回测截止日之后才会成交的末尾信号。"""
    if price == "same_close":
        return [day for day in signals if day <= end]
    positions = {day: index for index, day in enumerate(calendar)}
    return [
        day
        for day in signals
        if positions[day] + 1 < len(calendar) and calendar[positions[day] + 1] <= end
    ]


def _execution_setup(config: Config) -> tuple[ExecutionConfig, Any, Any, Any]:
    profile = str(config.rules.get("profile", "standard"))
    if profile not in {"minimal", "standard", "custom"}:
        raise ValueError("rules.profile 只能是 minimal、standard 或 custom")
    rules = dict(config.rules)
    if profile == "minimal":
        rules.update(
            block_suspended=False,
            block_limit=False,
            t_plus_one=False,
        )
        costs: Any = HistoricalAshareCosts(0.0, 0.0, False, False)
    else:
        method = config.costs.get("method")
        costs = import_callable(str(method)) if method else HistoricalAshareCosts.from_dict(config.costs)
    trade_rule = config.rules.get("method")
    lot_method = config.execution.get("lot_rounder")
    lot_rounder = import_callable(str(lot_method)) if lot_method else None
    return (
        ExecutionConfig.from_dict(config.execution, rules, profile=profile),
        costs,
        trade_rule,
        lot_rounder,
    )


def _apply_empty_signal_policy(
    targets: pl.DataFrame,
    panel: pl.DataFrame,
    signal_dates: list[date],
    policy: str,
) -> pl.DataFrame:
    """明确处理整期无有效分数，避免无声地沿用旧仓位。"""
    if policy not in {"error", "hold", "cash"}:
        raise ValueError("portfolio.empty_signal 只能是 error、hold 或 cash")
    compact = targets.select("date", "stock", "target_weight")
    present = set(compact["date"].unique().to_list())
    missing = sorted(set(signal_dates) - present)
    if not missing:
        return compact
    shown = ", ".join(map(str, missing[:3]))
    suffix = "..." if len(missing) > 3 else ""
    if policy == "error":
        raise ValueError(f"调仓日没有有效分数: {shown}{suffix}")
    if policy == "hold":
        return compact

    placeholders = (
        panel.filter(pl.col("date").is_in(missing))
        .group_by("date")
        .agg(pl.col("stock").sort().first())
        .with_columns(pl.lit(0.0).alias("target_weight"))
    )
    found = set(placeholders["date"].to_list())
    if found != set(missing):
        raise ValueError("empty_signal=cash 时至少需要一只可识别的成分股")
    return pl.concat([compact, placeholders], how="vertical").sort(["date", "stock"])


def _write_factor_reports(
    reports: list[FactorReport],
    config: Config,
    specs: tuple[FactorSpec, ...],
) -> Path:
    output = Path(config.output["directory"]) / "single_factor"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for report in reports:
        report.write_csv(output)
        summaries.append(report.summary)
    pl.DataFrame(summaries).write_csv(output / "summary.csv")
    (output / "resolved_config.json").write_text(
        _resolved_config_json(config, specs), encoding="utf-8"
    )
    return output


def _write_selection(
    result: SelectionResult,
    config: Config,
    specs: tuple[FactorSpec, ...],
) -> Path:
    output = Path(config.output["directory"]) / "factor_selection"
    output.mkdir(parents=True, exist_ok=True)
    metadata = pl.DataFrame(
        {
            "factor": [spec.output_name for spec in specs],
            "source": [spec.source for spec in specs],
            "raw_name": [spec.name for spec in specs],
        }
    )
    table = result.table.join(metadata, on="factor", how="left").select(
        "factor", "source", "raw_name", *result.table.columns[1:]
    )
    table.write_csv(output / "selection.csv")
    (output / "selected_factors.txt").write_text(
        "\n".join(result.selected), encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        _resolved_config_json(config, specs), encoding="utf-8"
    )
    return output


def _write_validation(
    result: ValidationResult,
    config: Config,
    specs: tuple[FactorSpec, ...],
) -> Path:
    output = Path(config.output["directory"]) / "factor_validation"
    output.mkdir(parents=True, exist_ok=True)
    metadata = pl.DataFrame(
        {
            "factor": [spec.output_name for spec in specs],
            "source": [spec.source for spec in specs],
            "raw_name": [spec.name for spec in specs],
        }
    )
    factors = result.factor_table.join(metadata, on="factor", how="left").select(
        "factor", "source", "raw_name", *result.factor_table.columns[1:]
    )
    result.split_table.write_csv(output / "split.csv")
    result.fold_table.write_csv(output / "folds.csv")
    result.candidate_table.write_csv(output / "candidates.csv")
    result.stability_table.write_csv(output / "factor_stability.csv")
    factors.write_csv(output / "factor_results.csv")
    factors.filter(pl.col("selected")).write_csv(output / "selected_factors.csv")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "validation_method": result.method,
                "best_ic_threshold": result.best_ic_threshold,
                "best_lasso_alpha": result.best_alpha,
                "selected_count": len(result.selected),
                "selected_factors": list(result.selected),
                "test_and_live_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "selected_factors.txt").write_text(
        "\n".join(result.selected), encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        _resolved_config_json(config, specs), encoding="utf-8"
    )
    return output


def _write_backtest(
    config: Config,
    strategy: BacktestResult,
    benchmark: BacktestResult,
    comparison: pl.DataFrame,
    targets: pl.DataFrame,
    panel: pl.DataFrame,
    summary: dict[str, Any],
    specs: tuple[FactorSpec, ...],
) -> Path:
    output = Path(config.output["directory"]) / "backtest"
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
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        _resolved_config_json(config, specs), encoding="utf-8"
    )
    return output


def _resolved_config_json(
    config: Config, specs: tuple[FactorSpec, ...]
) -> str:
    """把运行时补全的 factor source 一并写入，保证配置可复现。"""
    value = config.effective_dict()
    value["factors"] = [asdict(spec) for spec in specs]
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
