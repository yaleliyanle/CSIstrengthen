"""Factor selection chooses factors; cleaning, combination, and backtesting are handled elsewhere."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from .plugins import import_callable


@dataclass(frozen=True)
class SelectionResult:
    method: str
    forward_return: str
    table: pl.DataFrame
    selected: tuple[str, ...]


@dataclass(frozen=True)
class LassoModel:
    """Training fit; apply it to validation data without refitting."""

    columns: tuple[str, ...]
    coefficients: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    coverage: np.ndarray
    usable: np.ndarray
    selected: np.ndarray
    observations: int
    iterations: int
    converged: bool


def select_factors(
    panel: pl.DataFrame,
    factor_columns: Sequence[str],
    return_column: str,
    cfg: Mapping[str, Any] | Any,
) -> SelectionResult:
    """Select neutralized and standardized factors using the configuration."""

    options = _section(cfg)
    method = str(options.get("method", "ic"))
    columns = list(dict.fromkeys(factor_columns))
    _require_columns(panel, ["date", return_column, *columns])
    if not columns:
        raise ValueError("At least one candidate factor is required")

    if ":" in method:
        table = import_callable(method)(panel, columns, return_column, cfg)
        if not isinstance(table, pl.DataFrame):
            raise TypeError("Custom selection methods must return a Polars DataFrame")
        _require_columns(table, ["factor", "selected"])
    elif method == "ic":
        table = _ic_selection(panel, columns, return_column, options)
    elif method == "lasso":
        table = _lasso_selection(panel, columns, return_column, options)
    else:
        raise ValueError(f"Unknown factor selection method: {method!r}")

    table = table.with_columns(pl.col("selected").cast(pl.Boolean, strict=True))
    selected = tuple(
        table.filter(pl.col("selected")).get_column("factor").to_list()
    )
    return SelectionResult(method, return_column, table, selected)


def _ic_selection(
    panel: pl.DataFrame,
    columns: list[str],
    return_column: str,
    options: Mapping[str, Any],
) -> pl.DataFrame:
    min_coverage = float(options.get("min_coverage", 0.8))
    min_rank_ic = float(options.get("min_abs_rank_ic", 0.02))
    min_t = float(options.get("min_abs_t", 2.0))
    rows = []
    for column in columns:
        pairs = _finite_pairs(panel, column, return_column).with_columns(
            pl.col(column).rank("average").over("date").alias("__factor_rank"),
            pl.col(return_column).rank("average").over("date").alias("__return_rank"),
        )
        daily = pairs.group_by("date").agg(
            pl.corr(column, return_column).alias("ic"),
            pl.corr("__factor_rank", "__return_rank").alias("rank_ic"),
        )
        ic = _stats(daily.get_column("ic"))
        rank_ic = _stats(daily.get_column("rank_ic"))
        coverage = _coverage(panel, column)
        selected = (
            coverage >= min_coverage
            and abs(rank_ic["mean"]) >= min_rank_ic
            and abs(rank_ic["t_value"]) >= min_t
        )
        rows.append(
            {
                "factor": _short_name(column),
                "coverage": coverage,
                "observations": pairs.height,
                "dates": rank_ic["count"],
                "ic_mean": ic["mean"],
                "rank_ic_mean": rank_ic["mean"],
                "rank_ic_t": rank_ic["t_value"],
                "suggested_direction": 1 if rank_ic["mean"] >= 0 else -1,
                "selected": bool(selected),
            }
        )
    return pl.DataFrame(rows).sort(
        ["selected", "rank_ic_mean"], descending=[True, True]
    )


def _lasso_selection(
    panel: pl.DataFrame,
    columns: list[str],
    return_column: str,
    options: Mapping[str, Any],
) -> pl.DataFrame:
    """Solve standardized Lasso by coordinate descent; fill missing factors with neutral value zero."""

    model = fit_lasso_model(panel, columns, return_column, options)
    return lasso_table(model)


def factor_statistics(
    panel: pl.DataFrame,
    columns: Sequence[str],
    return_column: str,
) -> pl.DataFrame:
    """Return single-factor IC, coverage, and direction statistics without thresholds."""

    return _ic_selection(
        panel,
        list(columns),
        return_column,
        {"min_coverage": 0.0, "min_abs_rank_ic": 0.0, "min_abs_t": 0.0},
    ).drop("selected")


def fit_lasso_model(
    panel: pl.DataFrame,
    columns: Sequence[str],
    return_column: str,
    options: Mapping[str, Any],
) -> LassoModel:
    """Fit Lasso on the supplied training data only and retain scaling parameters for validation."""

    columns = list(columns)
    _require_columns(panel, ["date", return_column, *columns])
    alpha = float(options.get("alpha", 0.02))
    tolerance = float(options.get("tolerance", 1e-7))
    max_iter = int(options.get("max_iter", 1000))
    min_coverage = float(options.get("min_coverage", 0.8))
    coefficient_tolerance = float(options.get("coefficient_tolerance", 1e-10))
    top_n = int(options.get("top_n", 0))
    if alpha < 0 or tolerance <= 0 or max_iter < 1 or top_n < 0:
        raise ValueError("Lasso requires alpha>=0, tolerance>0, max_iter>=1, and top_n>=0")

    clean = panel.with_columns(
        (pl.col(return_column) - pl.col(return_column).mean().over("date")).alias(
            "__target"
        )
    ).filter(pl.col("__target").is_finite())
    if clean.is_empty():
        raise ValueError("No available forward returns in the selection interval")

    x = clean.select(columns).to_numpy().astype(float, copy=False)
    y = clean.get_column("__target").to_numpy().astype(float, copy=False)
    finite = np.isfinite(x)
    coverage = finite.mean(axis=0)
    means = np.nanmean(np.where(finite, x, np.nan), axis=0)
    scales = np.nanstd(np.where(finite, x, np.nan), axis=0)
    usable = (coverage >= min_coverage) & np.isfinite(scales) & (scales > 1e-12)
    x = np.where(finite, (x - means) / np.where(scales > 0, scales, 1.0), 0.0)
    x[:, ~usable] = 0.0
    y = y - y.mean()
    y_scale = y.std()
    if not np.isfinite(y_scale) or y_scale <= 1e-14:
        raise ValueError("Forward returns have no cross-sectional variation; cannot run Lasso")
    y = y / y_scale

    coefficients, iterations, converged = _coordinate_descent(
        x, y, alpha, max_iter, tolerance
    )
    selected = usable & (np.abs(coefficients) > coefficient_tolerance)
    if top_n and selected.sum() > top_n:
        keep = np.argsort(np.abs(coefficients))[-top_n:]
        selected[:] = False
        selected[keep] = usable[keep]

    return LassoModel(
        columns=tuple(columns),
        coefficients=coefficients,
        means=means,
        scales=scales,
        coverage=coverage,
        usable=usable,
        selected=selected,
        observations=clean.height,
        iterations=iterations,
        converged=converged,
    )


def lasso_table(model: LassoModel) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "factor": [_short_name(column) for column in model.columns],
            "coverage": model.coverage,
            "observations": [model.observations] * len(model.columns),
            "coefficient": model.coefficients,
            "abs_coefficient": np.abs(model.coefficients),
            "suggested_direction": np.where(model.coefficients >= 0, 1, -1),
            "selected": model.selected,
            "iterations": [model.iterations] * len(model.columns),
            "converged": [model.converged] * len(model.columns),
        }
    ).sort(["selected", "abs_coefficient"], descending=True)


def apply_lasso(panel: pl.DataFrame, model: LassoModel) -> pl.DataFrame:
    """Score validation data using the training model without accessing validation returns."""

    _require_columns(panel, ["date", "stock", *model.columns])
    x = panel.select(model.columns).to_numpy().astype(float, copy=False)
    finite = np.isfinite(x)
    x = np.where(
        finite,
        (x - model.means) / np.where(model.scales > 0, model.scales, 1.0),
        0.0,
    )
    x[:, ~model.usable] = 0.0
    return panel.with_columns(pl.Series("selection_score", x @ model.coefficients))


def score_statistics(
    panel: pl.DataFrame,
    score_column: str,
    return_column: str,
) -> dict[str, float | int]:
    """Summarize the combined score's RankIC across periods."""

    pairs = _finite_pairs(panel, score_column, return_column).with_columns(
        pl.col(score_column).rank("average").over("date").alias("__score_rank"),
        pl.col(return_column).rank("average").over("date").alias("__return_rank"),
    )
    daily = pairs.group_by("date").agg(
        pl.corr("__score_rank", "__return_rank").alias("rank_ic")
    )
    return _stats(daily.get_column("rank_ic"))


def _coordinate_descent(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, int, bool]:
    """Solve 1/(2n)||y-Xb||² + alpha*|b|."""

    n, features = x.shape
    beta = np.zeros(features)
    residual = y.copy()
    denominators = (x * x).sum(axis=0) / n
    for iteration in range(1, max_iter + 1):
        largest_change = 0.0
        for index in range(features):
            if denominators[index] <= 1e-14:
                continue
            old = beta[index]
            residual += x[:, index] * old
            rho = float(x[:, index] @ residual) / n
            beta[index] = _soft_threshold(rho, alpha) / denominators[index]
            residual -= x[:, index] * beta[index]
            largest_change = max(largest_change, abs(beta[index] - old))
        if largest_change < tolerance:
            return beta, iteration, True
    return beta, max_iter, False


def _soft_threshold(value: float, penalty: float) -> float:
    return np.sign(value) * max(abs(value) - penalty, 0.0)


def _finite_pairs(
    panel: pl.DataFrame, factor: str, forward_return: str
) -> pl.DataFrame:
    return panel.select("date", factor, forward_return).filter(
        pl.col(factor).is_finite() & pl.col(forward_return).is_finite()
    )


def _coverage(panel: pl.DataFrame, column: str) -> float:
    return float(panel.select(pl.col(column).is_finite().mean()).item() or 0.0)


def _stats(series: pl.Series) -> dict[str, float | int]:
    values = series.cast(pl.Float64, strict=False).drop_nulls().to_numpy()
    values = values[np.isfinite(values)]
    count = len(values)
    if not count:
        return {"count": 0, "mean": 0.0, "t_value": 0.0}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    if std > 1e-14:
        t_value = float(mean / (std / np.sqrt(count)))
    else:
        t_value = float("inf") if abs(mean) > 1e-14 else 0.0
    return {"count": count, "mean": mean, "t_value": t_value}


def _short_name(column: str) -> str:
    return column.removeprefix("factor__")


def _section(cfg: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if hasattr(cfg, "selection"):
        return cfg.selection
    if isinstance(cfg, Mapping):
        value = cfg.get("selection")
        return value if isinstance(value, Mapping) else cfg
    raise TypeError("cfg must be Config or dict")


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


__all__ = [
    "LassoModel",
    "SelectionResult",
    "apply_lasso",
    "factor_statistics",
    "fit_lasso_model",
    "lasso_table",
    "score_statistics",
    "select_factors",
]
