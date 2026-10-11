"""Convert combined scores into target weights.

For custom stock selection, implement ``builder(panel, config) -> DataFrame`` and configure
``method = "my_module:my_builder"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .plugins import resolve_method


KEYS = ["date", "stock"]


def top_n_equal(panel: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Buy the highest-scoring N stocks each period with equal weights."""
    top = _top_n(panel, int(config.get("top_n", 100)))
    return top.with_columns(
        (1.0 / pl.len().over("date")).alias("target_weight")
    )


def top_n_benchmark(panel: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Select the highest-scoring N stocks and renormalize their index weights."""
    top = _top_n(panel, int(config.get("top_n", 100)))
    total = pl.col("benchmark_weight").sum().over("date")
    return top.with_columns(
        pl.when(total > 0)
        .then(pl.col("benchmark_weight") / total)
        .otherwise(1.0 / pl.len().over("date"))
        .alias("target_weight")
    )


def benchmark_tilt(panel: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Multiply index weights by ``exp(tilt * score)`` for a moderate index-enhancement tilt."""
    tilt = float(config.get("tilt", 0.35))
    score_clip = float(config.get("score_clip", 3.0))
    frame = _valid(panel).with_columns(
        (
            pl.col("benchmark_weight").clip(lower_bound=0.0)
            * (pl.col("score").clip(-score_clip, score_clip) * tilt).exp()
        ).alias("_raw_weight")
    )
    frame = frame.with_columns(
        (pl.col("_raw_weight") / pl.col("_raw_weight").sum().over("date"))
        .fill_nan(0.0)
        .alias("target_weight")
    ).drop("_raw_weight")
    return _apply_cap(frame, float(config.get("max_weight", 0.0)))


BUILTINS = {
    "top_n_equal": top_n_equal,
    "top_n_benchmark": top_n_benchmark,
    "benchmark_tilt": benchmark_tilt,
}


def build_targets(panel: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Run the weight builder and validate long-only weights for each period."""
    required = set(KEYS + ["score", "benchmark_weight"])
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Portfolio input is missing columns: {sorted(missing)}")
    method = resolve_method(str(config.get("method", "benchmark_tilt")), BUILTINS)
    result = method(panel, config).sort(KEYS)
    required_output = set(KEYS + ["target_weight"])
    if not required_output <= set(result.columns):
        raise ValueError("Portfolio methods must return date, stock, target_weight")
    if result.filter(pl.col("target_weight") < -1e-12).height:
        raise ValueError("The engine accepts only nonnegative target weights")
    sums = result.group_by("date").agg(pl.col("target_weight").sum().alias("weight"))
    bad = sums.filter((pl.col("weight") - 1.0).abs() > 1e-6)
    if bad.height:
        raise ValueError(f"Target weights are not normalized; examples: {bad.head(3).to_dicts()}")
    return result


def _valid(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.filter(
        pl.col("score").is_finite()
        & pl.col("benchmark_weight").is_not_null()
        & (pl.col("benchmark_weight") >= 0)
    )


def _top_n(panel: pl.DataFrame, count: int) -> pl.DataFrame:
    if count < 1:
        raise ValueError("portfolio.top_n must be positive")
    return (
        _valid(panel)
        .with_columns(
            pl.col("score").rank("ordinal", descending=True).over("date").alias("_rank")
        )
        .filter(pl.col("_rank") <= count)
        .drop("_rank")
    )


def _apply_cap(frame: pl.DataFrame, cap: float) -> pl.DataFrame:
    if cap <= 0:
        return frame
    groups = [_cap_one_date(group, cap) for group in frame.partition_by("date")]
    return pl.concat(groups) if groups else frame


def _cap_one_date(frame: pl.DataFrame, cap: float) -> pl.DataFrame:
    weights = frame["target_weight"].to_numpy().astype(float)
    if cap * len(weights) < 1.0 - 1e-10:
        raise ValueError(f"max_weight={cap} is too small for {len(weights)} stocks to have weights summing to 1")
    result = np.zeros_like(weights)
    free = np.ones(len(weights), dtype=bool)
    remaining = 1.0
    while free.any():
        raw = weights[free]
        scaled = np.full(raw.size, remaining / raw.size) if raw.sum() <= 0 else raw / raw.sum() * remaining
        over = scaled > cap + 1e-14
        indices = np.flatnonzero(free)
        if not over.any():
            result[indices] = scaled
            break
        capped = indices[over]
        result[capped] = cap
        free[capped] = False
        remaining -= cap * len(capped)
    return frame.with_columns(pl.Series("target_weight", result))
