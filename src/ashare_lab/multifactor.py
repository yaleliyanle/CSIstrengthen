"""滚动多因子合成。

内置方法只决定怎样把多个已预处理因子合成 ``score``。股票池、Top N、
成交规则和费用仍由原有模块负责。自定义非线性/机器学习方法可以替换本模块
的整套打分入口，而不需要修改回测引擎。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

from .plugins import import_callable
from .settings import FactorSpec


@dataclass(frozen=True)
class MultiFactorScores:
    """动态合成的标准输出。"""

    method: str
    scores: pl.DataFrame
    factor_weights: pl.DataFrame
    ic_history: pl.DataFrame
    diagnostics: pl.DataFrame


def build_multifactor_scores(
    estimation_panel: pl.DataFrame,
    application_panel: pl.DataFrame,
    specs: Sequence[FactorSpec],
    cfg: Mapping[str, Any] | Any,
) -> MultiFactorScores:
    """用全A历史估计模型，并在目标指数股票池上生成综合分。"""

    options = _options(cfg)
    method = str(options.get("method", "max_ic_ir"))
    if ":" in method:
        result = import_callable(method)(
            estimation_panel, application_panel, specs, cfg
        )
        if isinstance(result, MultiFactorScores):
            return result
        if isinstance(result, pl.DataFrame):
            _require(result, ["date", "stock", "score"])
            return MultiFactorScores(
                method, result, pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
            )
        raise TypeError(
            "自定义 multifactor.method 必须返回 MultiFactorScores 或带 "
            "date/stock/score 的 Polars DataFrame"
        )
    if method not in {"equal", "max_ic_ir", "max_ic"}:
        raise ValueError(f"未知多因子合成方法: {method!r}")

    columns = [f"factor__{spec.output_name}" for spec in specs]
    required_history = ["date", "stock", "forward_return", "exit_date", *columns]
    _require(estimation_panel, required_history)
    _require(application_panel, ["date", "stock", "benchmark_weight", *columns])

    ic_history = factor_ic_history(estimation_panel, columns)
    return _rolling_linear_scores(
        method,
        estimation_panel,
        application_panel,
        specs,
        columns,
        ic_history,
        options,
    )


def factor_ic_history(
    panel: pl.DataFrame,
    factor_columns: Sequence[str],
    *,
    min_observations: int = 30,
) -> pl.DataFrame:
    """计算逐期全A RankIC；方向已由预处理阶段统一。"""

    rows: list[pl.DataFrame] = []
    for column in factor_columns:
        pairs = panel.select(
            "date", "exit_date", column, "forward_return"
        ).filter(
            pl.col(column).is_finite() & pl.col("forward_return").is_finite()
        ).with_columns(
            pl.col(column).rank("average").over("date").alias("_factor_rank"),
            pl.col("forward_return")
            .rank("average")
            .over("date")
            .alias("_return_rank"),
        )
        daily = pairs.group_by("date").agg(
            pl.col("exit_date").first(),
            pl.len().alias("observations"),
            pl.corr("_factor_rank", "_return_rank").alias("rank_ic"),
        ).with_columns(
            pl.lit(column.removeprefix("factor__")).alias("factor"),
            pl.when(
                (pl.col("observations") >= min_observations)
                & pl.col("rank_ic").is_finite()
            )
            .then(pl.col("rank_ic"))
            .otherwise(None)
            .alias("rank_ic"),
        )
        rows.append(daily)
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows).select(
        "date", "exit_date", "factor", "observations", "rank_ic"
    ).sort("date", "factor")


def solve_long_only_ratio(
    expected_ic: np.ndarray,
    risk_matrix: np.ndarray,
    *,
    eigenvalue_floor: float = 1e-8,
    tolerance: float = 1e-9,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """最大化 ``w'mu / sqrt(w'Aw)``，约束 ``w>=0, sum(w)=1``。"""

    mu = np.asarray(expected_ic, dtype=float).reshape(-1)
    risk = _sanitize_matrix(risk_matrix, eigenvalue_floor)
    if not len(mu) or risk.shape != (len(mu), len(mu)):
        raise ValueError("预期 IC 与风险矩阵维度不一致")
    if not np.isfinite(mu).all():
        raise ValueError("预期 IC 含非有限值")
    if len(mu) == 1:
        predicted_risk = math.sqrt(float(risk[0, 0]))
        return np.array([1.0]), {
            "predicted_ic": float(mu[0]),
            "predicted_risk": predicted_risk,
            "predicted_ratio": float(mu[0] / predicted_risk),
            "solver_success": True,
        }

    def ratio(weight: np.ndarray) -> float:
        variance = float(weight @ risk @ weight)
        if variance <= 0 or not np.isfinite(variance):
            return -math.inf
        return float(weight @ mu / math.sqrt(variance))

    def objective(weight: np.ndarray) -> float:
        value = ratio(weight)
        return -value if np.isfinite(value) else 1e12

    constraints = ({"type": "eq", "fun": lambda weight: weight.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * len(mu)
    starts = [np.full(len(mu), 1.0 / len(mu)), *np.eye(len(mu))]
    candidates: list[tuple[float, np.ndarray, Any]] = []
    for start in starts:
        solved = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": tolerance, "maxiter": max_iterations},
        )
        weight = np.clip(np.asarray(solved.x, dtype=float), 0.0, 1.0)
        if weight.sum() > 0:
            weight /= weight.sum()
        value = ratio(weight)
        if np.isfinite(value) and abs(weight.sum() - 1.0) <= 1e-7:
            candidates.append((value, weight, solved))
    if not candidates:
        raise RuntimeError("多因子权重优化没有得到可行解")
    value, weight, solved = max(candidates, key=lambda item: item[0])
    predicted_risk = math.sqrt(float(weight @ risk @ weight))
    return weight, {
        "predicted_ic": float(weight @ mu),
        "predicted_risk": predicted_risk,
        "predicted_ratio": value,
        "solver_success": bool(solved.success),
    }


def _rolling_linear_scores(
    method: str,
    estimation: pl.DataFrame,
    application: pl.DataFrame,
    specs: Sequence[FactorSpec],
    columns: list[str],
    ic_history: pl.DataFrame,
    options: Mapping[str, Any],
) -> MultiFactorScores:
    lookback = int(options.get("lookback_periods", 12))
    group_by_source = bool(options.get("group_by_source", True))
    shrinkage = float(options.get("exposure_shrinkage", 0.1))
    floor = float(options.get("eigenvalue_floor", 1e-8))
    tolerance = float(options.get("optimizer_tolerance", 1e-9))
    iterations = int(options.get("optimizer_max_iterations", 1000))

    groups: dict[str, list[str]] = {}
    for spec, column in zip(specs, columns, strict=True):
        group = str(spec.source or "未分类") if group_by_source else "全部因子"
        groups.setdefault(group, []).append(column)

    pivot = ic_history.pivot(
        on="factor", index=["date", "exit_date"], values="rank_ic"
    ).sort("date")
    factor_names = [column.removeprefix("factor__") for column in columns]
    for name in factor_names:
        if name not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    complete = pivot.filter(
        pl.all_horizontal([pl.col(name).is_not_null() for name in factor_names])
    )

    score_frames: list[pl.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    signal_dates = application.get_column("date").unique().sort().to_list()
    for signal_date in signal_dates:
        known = complete.filter(pl.col("exit_date") <= signal_date).tail(lookback)
        if known.height < lookback:
            continue
        current_all = estimation.filter(pl.col("date") == signal_date)
        current_target = application.filter(pl.col("date") == signal_date)
        if current_all.is_empty() or current_target.is_empty():
            continue

        category_scores: list[np.ndarray] = []
        for group, group_columns in groups.items():
            names = [name.removeprefix("factor__") for name in group_columns]
            history = known.select(names).to_numpy().astype(float, copy=False)
            expected_ic = history.mean(axis=0)
            current_exposure = _matrix(current_all, group_columns)
            if method == "equal":
                weight = np.full(len(group_columns), 1.0 / len(group_columns))
                risk = np.eye(len(group_columns))
                details = {
                    "predicted_ic": float(weight @ expected_ic),
                    "predicted_risk": 1.0,
                    "predicted_ratio": float(weight @ expected_ic),
                    "solver_success": True,
                }
            else:
                risk = (
                    np.atleast_2d(np.cov(history, rowvar=False, ddof=1))
                    if method == "max_ic_ir"
                    else _exposure_correlation(current_exposure, shrinkage)
                )
                weight, details = solve_long_only_ratio(
                    expected_ic,
                    risk,
                    eigenvalue_floor=floor,
                    tolerance=tolerance,
                    max_iterations=iterations,
                )
            raw_score = _matrix(current_target, group_columns) @ weight
            category_scores.append(_zscore(raw_score))
            for column, value, mean_ic in zip(
                group_columns, weight, expected_ic, strict=True
            ):
                weight_rows.append(
                    {
                        "date": signal_date,
                        "group": group,
                        "factor": column.removeprefix("factor__"),
                        "weight_within_group": float(value),
                        "weight_in_composite": float(value / len(groups)),
                        "mean_historical_rank_ic": float(mean_ic),
                        "lookback_start": known["date"][0],
                        "lookback_end": known["date"][-1],
                        "latest_known_exit": known["exit_date"][-1],
                    }
                )
            diagnostic_rows.append(
                {
                    "date": signal_date,
                    "group": group,
                    "method": method,
                    "factor_count": len(group_columns),
                    "maximum_factor_weight": float(weight.max()),
                    "effective_factor_count": float(1.0 / np.sum(weight**2)),
                    **details,
                }
            )
        score = np.mean(np.column_stack(category_scores), axis=1)
        score_frames.append(
            current_target.select("date", "stock", "benchmark_weight").with_columns(
                pl.Series("score", score)
            )
        )

    scores = (
        pl.concat(score_frames).sort("date", "stock")
        if score_frames
        else application.head(0).select("date", "stock", "benchmark_weight").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("score")
        )
    )
    weights = pl.DataFrame(weight_rows) if weight_rows else pl.DataFrame()
    diagnostics = pl.DataFrame(diagnostic_rows) if diagnostic_rows else pl.DataFrame()
    return MultiFactorScores(method, scores, weights, ic_history, diagnostics)


def _matrix(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame.select(columns).to_numpy().astype(float, copy=False)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    standard_deviation = float(values.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-14:
        return np.zeros(len(values))
    return (values - values.mean()) / standard_deviation


def _exposure_correlation(values: np.ndarray, shrinkage: float) -> np.ndarray:
    if values.shape[1] == 1:
        return np.ones((1, 1))
    standardized = np.column_stack(
        [_zscore(values[:, index]) for index in range(values.shape[1])]
    )
    correlation = np.atleast_2d(np.corrcoef(standardized, rowvar=False))
    return (1.0 - shrinkage) * correlation + shrinkage * np.eye(values.shape[1])


def _sanitize_matrix(matrix: np.ndarray, eigenvalue_floor: float) -> np.ndarray:
    result = np.asarray(matrix, dtype=float)
    if result.ndim == 0:
        result = result.reshape(1, 1)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError("风险矩阵必须是方阵")
    if not np.isfinite(result).all():
        raise ValueError("风险矩阵含非有限值")
    result = (result + result.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(result)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    eigenvalues = np.maximum(eigenvalues, eigenvalue_floor * scale)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _options(cfg: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if hasattr(cfg, "multifactor"):
        return cfg.multifactor
    if isinstance(cfg, Mapping):
        value = cfg.get("multifactor")
        return value if isinstance(value, Mapping) else cfg
    raise TypeError("cfg 应为 Config 或 dict")


def _require(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"多因子输入缺少列: {missing}")


__all__ = [
    "MultiFactorScores",
    "build_multifactor_scores",
    "factor_ic_history",
    "solve_long_only_ratio",
]
