"""两层因子筛选的固定划分与滚动验证。"""

from __future__ import annotations

import calendar as month_calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from .selection import (
    apply_lasso,
    factor_statistics,
    fit_lasso_model,
    lasso_table,
    score_statistics,
)


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date


@dataclass(frozen=True)
class ValidationFold:
    name: str
    train: Period
    validation: Period


@dataclass(frozen=True)
class ValidationResult:
    method: str
    best_ic_threshold: float
    best_alpha: float
    selected: tuple[str, ...]
    split_table: pl.DataFrame
    fold_table: pl.DataFrame
    candidate_table: pl.DataFrame
    stability_table: pl.DataFrame
    factor_table: pl.DataFrame


def dataset_split(cfg: Any) -> tuple[Period, Period, Period, Period]:
    """读取四段时间；后两段只登记，不参与本阶段计算。"""

    values = cfg.split if hasattr(cfg, "split") else cfg["split"]
    periods = (
        Period("train", _day(values["train_start"]), _day(values["train_end"])),
        Period(
            "validation",
            _day(values["validation_start"]),
            _day(values["validation_end"]),
        ),
        Period("test", _day(values["test_start"]), _day(values["test_end"])),
        Period("live", _day(values["live_start"]), _day(values["live_end"])),
    )
    for item in periods:
        if item.start > item.end:
            raise ValueError(f"{item.name} 的开始日不能晚于结束日")
    for left, right in zip(periods, periods[1:]):
        if left.end >= right.start:
            raise ValueError(f"{left.name} 与 {right.name} 时间重叠")
    return periods


def validation_folds(cfg: Any) -> list[ValidationFold]:
    """固定模式返回一折；滚动模式只在 train+validation 内生成折。"""

    train, validation, _, _ = dataset_split(cfg)
    split = cfg.split if hasattr(cfg, "split") else cfg["split"]
    method = str(split.get("method", "fixed"))
    if method == "fixed":
        return [ValidationFold("fixed_1", train, validation)]
    if method != "rolling":
        raise ValueError("split.method 只能是 fixed 或 rolling")

    rolling = cfg.rolling if hasattr(cfg, "rolling") else cfg["rolling"]
    train_months = int(rolling.get("train_months", 24))
    validation_months = int(rolling.get("validation_months", 6))
    step_months = int(rolling.get("step_months", 6))
    window = str(rolling.get("window", "expanding"))
    if min(train_months, validation_months, step_months) < 1:
        raise ValueError("滚动月份必须是正整数")
    if window not in {"expanding", "sliding"}:
        raise ValueError("rolling.window 只能是 expanding 或 sliding")

    folds = []
    index = 0
    while True:
        moving_start = _add_months(train.start, index * step_months)
        fold_train_start = train.start if window == "expanding" else moving_start
        train_boundary = _add_months(moving_start, train_months)
        validation_end = _add_months(train_boundary, validation_months) - timedelta(days=1)
        if validation_end > validation.end:
            break
        folds.append(
            ValidationFold(
                f"rolling_{index + 1}",
                Period("train", fold_train_start, train_boundary - timedelta(days=1)),
                Period("validation", train_boundary, validation_end),
            )
        )
        index += 1
    if not folds:
        raise ValueError("当前 rolling 月份在训练+验证区间内无法生成任何折")
    return folds


def purge_period(
    panel: pl.DataFrame,
    calendar: list[date],
    period: Period,
    horizon: int,
    execution: str,
) -> pl.DataFrame:
    """只保留前瞻收益终点仍落在本区间内的信号，防止跨界泄露。"""

    if execution not in {"same_close", "next_open"}:
        raise ValueError("execution 只能是 same_close 或 next_open")
    positions = {day: index for index, day in enumerate(calendar)}
    end_positions = [positions[day] for day in calendar if day <= period.end]
    if not end_positions:
        return panel.head(0)
    last = max(end_positions)
    start_offset = 1 if execution == "next_open" else 0
    dates = panel.get_column("date").unique().to_list()
    valid = [
        day
        for day in dates
        if period.start <= day <= period.end
        and day in positions
        and positions[day] + start_offset + horizon <= last
    ]
    return panel.filter(pl.col("date").is_in(valid))


def validate_two_stage(
    panel: pl.DataFrame,
    factor_columns: list[str],
    return_column: str,
    calendar: list[date],
    cfg: Any,
) -> ValidationResult:
    """第一层 IC 门槛，第二层 Lasso；参数只由验证期决定。"""

    selection = cfg.selection if hasattr(cfg, "selection") else cfg["selection"]
    split = cfg.split if hasattr(cfg, "split") else cfg["split"]
    execution_cfg = cfg.execution if hasattr(cfg, "execution") else cfg["execution"]
    method = str(split.get("method", "fixed"))
    horizon = int(selection.get("horizon", 20))
    execution = str(execution_cfg.get("price", "next_open"))
    ic_grid = [float(value) for value in selection.get("ic_thresholds", [0.01, 0.02, 0.03])]
    alpha_grid = [float(value) for value in selection.get("alpha_grid", [0.01, 0.02, 0.05])]
    min_coverage = float(selection.get("min_coverage", 0.8))
    min_t = float(selection.get("min_abs_t", 2.0))
    min_direction = float(selection.get("min_direction_stability", 0.6))
    if not ic_grid or not alpha_grid:
        raise ValueError("selection.ic_thresholds 和 alpha_grid 不能为空")

    folds = validation_folds(cfg)
    candidate_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = purge_period(panel, calendar, fold.train, horizon, execution)
        valid = purge_period(panel, calendar, fold.validation, horizon, execution)
        if train.is_empty() or valid.is_empty():
            raise ValueError(f"{fold.name} 清除跨界标签后没有足够数据")
        train_stats = factor_statistics(train, factor_columns, return_column)
        valid_stats = factor_statistics(valid, factor_columns, return_column)
        train_map = _rows_by_factor(train_stats)
        valid_map = _rows_by_factor(valid_stats)

        for threshold in ic_grid:
            passed = train_stats.filter(
                (pl.col("coverage") >= min_coverage)
                & (pl.col("rank_ic_mean").abs() >= threshold)
                & (pl.col("rank_ic_t").abs() >= min_t)
            ).get_column("factor").to_list()
            columns = [column for column in factor_columns if _short(column) in passed]
            for alpha in alpha_grid:
                row: dict[str, Any] = {
                    "fold": fold.name,
                    "ic_threshold": threshold,
                    "alpha": alpha,
                    "ic_passed": len(columns),
                    "lasso_selected": 0,
                    "validation_rank_ic": None,
                    "validation_rank_ic_t": None,
                    "direction_stability": None,
                }
                if not columns:
                    candidate_rows.append(row)
                    continue
                options = dict(selection)
                options["alpha"] = alpha
                model = fit_lasso_model(train, columns, return_column, options)
                chosen = [
                    _short(column)
                    for column, keep in zip(model.columns, model.selected)
                    if keep
                ]
                row["lasso_selected"] = len(chosen)
                if not chosen:
                    candidate_rows.append(row)
                    continue
                scored = apply_lasso(valid, model)
                score = score_statistics(scored, "selection_score", return_column)
                stable = [
                    _same_direction(train_map[name], valid_map[name])
                    for name in chosen
                ]
                row.update(
                    validation_rank_ic=score["mean"],
                    validation_rank_ic_t=score["t_value"],
                    direction_stability=float(np.mean(stable)),
                )
                candidate_rows.append(row)
                coefficient = dict(
                    zip(
                        [_short(column) for column in model.columns],
                        model.coefficients,
                    )
                )
                for name in chosen:
                    factor_rows.append(
                        {
                            "fold": fold.name,
                            "ic_threshold": threshold,
                            "alpha": alpha,
                            "factor": name,
                            "coefficient": coefficient[name],
                            "train_rank_ic": train_map[name]["rank_ic_mean"],
                            "validation_rank_ic": valid_map[name]["rank_ic_mean"],
                            "direction_stable": _same_direction(
                                train_map[name], valid_map[name]
                            ),
                        }
                    )

    candidates = pl.DataFrame(candidate_rows)
    summary = _candidate_summary(candidates, len(folds), min_direction)
    eligible = summary.filter(pl.col("eligible"))
    if eligible.is_empty():
        raise ValueError("没有参数组合通过方向稳定性要求；请查看候选结果或降低门槛")
    best = eligible.sort(
        ["validation_rank_ic_mean", "direction_stability_mean"],
        descending=True,
    ).row(0, named=True)
    best_ic = float(best["ic_threshold"])
    best_alpha = float(best["alpha"])

    # 参数锁定后，允许用 train+validation 重新估计；test/live 仍完全封存。
    train_period, validation_period, _, _ = dataset_split(cfg)
    development = Period("development", train_period.start, validation_period.end)
    full = purge_period(panel, calendar, development, horizon, execution)
    full_stats = factor_statistics(full, factor_columns, return_column)
    passed = full_stats.filter(
        (pl.col("coverage") >= min_coverage)
        & (pl.col("rank_ic_mean").abs() >= best_ic)
        & (pl.col("rank_ic_t").abs() >= min_t)
    ).get_column("factor").to_list()
    final_columns = [column for column in factor_columns if _short(column) in passed]
    if not final_columns:
        raise ValueError("训练+验证合并后没有因子通过最终 IC 门槛")
    options = dict(selection)
    options["alpha"] = best_alpha
    final_model = fit_lasso_model(full, final_columns, return_column, options)
    final_lasso = lasso_table(final_model)

    factor_history = pl.DataFrame(factor_rows) if factor_rows else _empty_history()
    chosen_history = factor_history.filter(
        (pl.col("ic_threshold") == best_ic) & (pl.col("alpha") == best_alpha)
    )
    stability = _factor_stability(chosen_history, len(folds))
    final_table = (
        final_lasso.join(full_stats, on="factor", how="left", suffix="_ic")
        .join(stability, on="factor", how="left")
        .sort(["selected", "abs_coefficient"], descending=True)
    )
    selected = tuple(
        final_table.filter(pl.col("selected")).get_column("factor").to_list()
    )
    return ValidationResult(
        method=method,
        best_ic_threshold=best_ic,
        best_alpha=best_alpha,
        selected=selected,
        split_table=_split_table(cfg),
        fold_table=_fold_table(folds),
        candidate_table=summary,
        stability_table=stability,
        factor_table=final_table,
    )


def _candidate_summary(
    candidates: pl.DataFrame, fold_count: int, min_direction: float
) -> pl.DataFrame:
    return (
        candidates.group_by("ic_threshold", "alpha")
        .agg(
            pl.col("validation_rank_ic").count().alias("valid_folds"),
            pl.col("validation_rank_ic").mean().alias("validation_rank_ic_mean"),
            pl.col("validation_rank_ic").min().alias("validation_rank_ic_worst"),
            (pl.col("validation_rank_ic") > 0).mean().alias("positive_fold_ratio"),
            pl.col("direction_stability").mean().alias("direction_stability_mean"),
            pl.col("ic_passed").mean().alias("mean_ic_passed"),
            pl.col("lasso_selected").mean().alias("mean_lasso_selected"),
        )
        .with_columns(
            (
                (pl.col("valid_folds") == fold_count)
                & (pl.col("validation_rank_ic_mean") > 0)
                & (pl.col("direction_stability_mean") >= min_direction)
            ).alias("eligible")
        )
        .sort("ic_threshold", "alpha")
    )


def _factor_stability(history: pl.DataFrame, fold_count: int) -> pl.DataFrame:
    if history.is_empty():
        return pl.DataFrame(
            schema={
                "factor": pl.String,
                "selection_frequency": pl.Float64,
                "direction_stability": pl.Float64,
                "mean_validation_rank_ic": pl.Float64,
            }
        )
    return history.group_by("factor").agg(
        (pl.len() / fold_count).alias("selection_frequency"),
        pl.col("direction_stable").mean().alias("direction_stability"),
        pl.col("validation_rank_ic").mean().alias("mean_validation_rank_ic"),
    )


def _split_table(cfg: Any) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"period": item.name, "start": item.start, "end": item.end}
            for item in dataset_split(cfg)
        ]
    )


def _fold_table(folds: list[ValidationFold]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "fold": fold.name,
                "train_start": fold.train.start,
                "train_end": fold.train.end,
                "validation_start": fold.validation.start,
                "validation_end": fold.validation.end,
            }
            for fold in folds
        ]
    )


def _rows_by_factor(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    return {row["factor"]: row for row in frame.iter_rows(named=True)}


def _same_direction(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return float(left["rank_ic_mean"]) * float(right["rank_ic_mean"]) > 0


def _empty_history() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "fold": pl.String,
            "ic_threshold": pl.Float64,
            "alpha": pl.Float64,
            "factor": pl.String,
            "coefficient": pl.Float64,
            "train_rank_ic": pl.Float64,
            "validation_rank_ic": pl.Float64,
            "direction_stable": pl.Boolean,
        }
    )


def _short(column: str) -> str:
    return column.removeprefix("factor__")


def _day(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(day.day, month_calendar.monthrange(year, month)[1]))


__all__ = [
    "Period",
    "ValidationFold",
    "ValidationResult",
    "dataset_split",
    "purge_period",
    "validate_two_stage",
    "validation_folds",
]
