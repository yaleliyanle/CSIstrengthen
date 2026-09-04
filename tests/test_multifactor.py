from datetime import date

import numpy as np
import polars as pl

from ashare_lab.multifactor import build_multifactor_scores, solve_long_only_ratio
from ashare_lab.multifactor_pipeline import period_forward_returns
from ashare_lab.settings import FactorSpec


def test_period_return_uses_next_rebalance_execution_prices():
    days = [date(2024, 1, day) for day in range(2, 7)]
    market = pl.DataFrame(
        {
            "date": days,
            "stock": ["A"] * len(days),
            "adj_open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "adj_close": [10.5, 11.5, 12.5, 13.5, 14.5],
        }
    )
    labels, periods = period_forward_returns(
        market, [days[0], days[2], days[4]], days, "next_open"
    )

    assert periods.row(0, named=True) == {
        "date": days[0],
        "entry_date": days[1],
        "exit_date": days[3],
    }
    assert np.isclose(labels["forward_return"][0], 13.0 / 11.0 - 1.0)


def test_dynamic_weights_only_use_completed_ic_periods():
    dates = [date(2024, month, 1) for month in range(1, 5)]
    exits = [date(2024, month + 1, 1) for month in range(1, 5)]
    stocks = [f"S{number:02d}" for number in range(40)]
    rows = []
    for signal, exit_date in zip(dates, exits, strict=True):
        for number, stock in enumerate(stocks):
            value = float(number)
            rows.append(
                {
                    "date": signal,
                    "stock": stock,
                    "factor__good": value,
                    "factor__bad": -value,
                    "forward_return": value / 100.0,
                    "exit_date": exit_date,
                }
            )
    estimation = pl.DataFrame(rows)
    application = estimation.select(
        "date", "stock", "factor__good", "factor__bad"
    ).with_columns(pl.lit(1.0 / len(stocks)).alias("benchmark_weight"))
    specs = (
        FactorSpec("good", source="demo"),
        FactorSpec("bad", source="demo"),
    )

    result = build_multifactor_scores(
        estimation,
        application,
        specs,
        {"multifactor": {"method": "max_ic_ir", "lookback_periods": 2}},
    )

    assert result.scores["date"].unique().sort().to_list() == dates[2:]
    assert result.factor_weights.filter(pl.col("date") == dates[2]).select(
        pl.col("latest_known_exit").max()
    ).item() <= dates[2]
    first_weights = result.factor_weights.filter(pl.col("date") == dates[2])
    assert first_weights.filter(pl.col("factor") == "good")[
        "weight_within_group"
    ][0] > 0.99


def test_ratio_optimizer_is_long_only_and_normalized():
    weights, _ = solve_long_only_ratio(
        np.array([0.05, 0.01]), np.array([[1.0, 0.0], [0.0, 1.0]])
    )
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]
