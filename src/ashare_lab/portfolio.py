"""把合成分数变成目标权重。

自定义选股只要实现 ``builder(panel, config) -> DataFrame``，并在配置中写
``method = "my_module:my_builder"``。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .plugins import resolve_method


KEYS = ["date", "stock"]


def top_n_equal(panel: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """每期买入分数最高的 N 只，等权。"""
    top = _top_n(panel, int(config.get("top_n", 100)))
    return top.with_columns(
        (1.0 / pl.len().over("date")).alias("target_weight")
    )


def top_n_benchmark(panel: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """最高 N 只，保留并重新归一化其指数权重。"""
    top = _top_n(panel, int(config.get("top_n", 100)))
    total = pl.col("benchmark_weight").sum().over("date")
    return top.with_columns(
        pl.when(total > 0)
        .then(pl.col("benchmark_weight") / total)
        .otherwise(1.0 / pl.len().over("date"))
        .alias("target_weight")
    )


def benchmark_tilt(panel: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """指数权重乘 ``exp(tilt * score)``，适合做温和指数增强。"""
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
    """运行权重生成器并检查每期多头权重。"""
    required = set(KEYS + ["score", "benchmark_weight"])
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"组合输入缺少列: {sorted(missing)}")
    method = resolve_method(str(config.get("method", "benchmark_tilt")), BUILTINS)
    result = method(panel, config).sort(KEYS)
    required_output = set(KEYS + ["target_weight"])
    if not required_output <= set(result.columns):
        raise ValueError("组合方法必须返回 date, stock, target_weight")
    if result.filter(pl.col("target_weight") < -1e-12).height:
        raise ValueError("当前引擎只接受非负目标权重")
    sums = result.group_by("date").agg(pl.col("target_weight").sum().alias("weight"))
    bad = sums.filter((pl.col("weight") - 1.0).abs() > 1e-6)
    if bad.height:
        raise ValueError(f"目标权重未归一化，示例: {bad.head(3).to_dicts()}")
    return result


def _valid(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.filter(
        pl.col("score").is_finite()
        & pl.col("benchmark_weight").is_not_null()
        & (pl.col("benchmark_weight") >= 0)
    )


def _top_n(panel: pl.DataFrame, count: int) -> pl.DataFrame:
    if count < 1:
        raise ValueError("portfolio.top_n 必须大于 0")
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
        raise ValueError(f"max_weight={cap} 太小，{len(weights)} 只股票无法使权重和为 1")
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
