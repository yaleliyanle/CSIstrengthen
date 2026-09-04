from datetime import date

import polars as pl

from ashare_lab.selection import select_factors


def test_lasso_keeps_known_signal_and_drops_noise():
    stocks = [f"S{i}" for i in range(20)]
    signal = [float(i - 10) for i in range(20)]
    noise = [float((i * 7) % 11 - 5) for i in range(20)]
    panel = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 20,
            "stock": stocks,
            "factor__signal": signal,
            "factor__noise": noise,
            "forward_5d": [3 * value for value in signal],
        }
    )

    result = select_factors(
        panel,
        ["factor__signal", "factor__noise"],
        "forward_5d",
        {"method": "lasso", "alpha": 0.1, "min_coverage": 1.0},
    )

    assert result.selected == ("signal",)
