"""最小插件示例。

运行时把 ``portfolio.method`` 改为
``examples.custom_methods:top_twenty_percent`` 即可，无需改框架代码。
"""

from typing import Any

import polars as pl


def top_twenty_percent(panel: pl.DataFrame, _: dict[str, Any]) -> pl.DataFrame:
    """每期取分数最高 20%，等权。"""
    ranked = panel.filter(pl.col("score").is_not_null()).with_columns(
        pl.col("score").rank(descending=True).over("date").alias("rank"),
        pl.len().over("date").alias("count"),
    )
    return (
        ranked.filter(pl.col("rank") <= (pl.col("count") * 0.2).ceil())
        .with_columns((1.0 / pl.len().over("date")).alias("target_weight"))
        .drop("rank", "count")
    )
