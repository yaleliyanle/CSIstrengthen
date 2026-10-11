"""Minimal plugin example.

Set ``portfolio.method`` to
``examples.custom_methods:top_twenty_percent`` without changing the framework.
"""

from typing import Any

import polars as pl


def top_twenty_percent(panel: pl.DataFrame, _: dict[str, Any]) -> pl.DataFrame:
    """Choose the highest-scoring 20% each period and weight them equally."""
    ranked = panel.filter(pl.col("score").is_not_null()).with_columns(
        pl.col("score").rank(descending=True).over("date").alias("rank"),
        pl.len().over("date").alias("count"),
    )
    return (
        ranked.filter(pl.col("rank") <= (pl.col("count") * 0.2).ceil())
        .with_columns((1.0 / pl.len().over("date")).alias("target_weight"))
        .drop("rank", "count")
    )
