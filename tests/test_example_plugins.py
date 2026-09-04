from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import polars as pl

from ashare_lab.execution import OrderContext, TradeCost
from ashare_lab.plugins import import_callable


MODULE = "ashare_lab.example_plugins"


def test_example_plugins_are_importable_and_follow_interfaces() -> None:
    portfolio = import_callable(f"{MODULE}:top_twenty_percent")
    combine = import_callable(f"{MODULE}:average_available_factors")
    schedule = import_callable(f"{MODULE}:first_trading_day_each_month")
    rule = import_callable(f"{MODULE}:block_new_stocks")
    costs = import_callable(f"{MODULE}:zero_cost")

    day = date(2024, 1, 2)
    panel = pl.DataFrame(
        {
            "date": [day] * 10,
            "stock": [f"S{index}" for index in range(10)],
            "score": list(range(10)),
        }
    )
    targets = portfolio(panel, {})
    assert targets.height == 2
    assert targets.get_column("target_weight").sum() == 1.0

    factors = panel.with_columns(
        pl.lit(1.0).alias("factor__quality"),
        pl.lit(3.0).alias("factor__value"),
    )
    score = combine(
        factors,
        [SimpleNamespace(output_name="quality"), SimpleNamespace(output_name="value")],
        {},
    )
    assert factors.select(score).item(0, "score") == 2.0

    calendar = [date(2024, 1, 2), date(2024, 1, 31), date(2024, 2, 1)]
    assert schedule(calendar, {}) == [date(2024, 1, 2), date(2024, 2, 1)]

    context = OrderContext(day, "IPO", "buy", 10.0, 10.0, "N", 0.0)
    assert rule(context) == "custom_new_stock"
    assert isinstance(costs(day, "buy", 100_000.0), TradeCost)
    assert costs(day, "buy", 100_000.0).total == 0.0
