from datetime import date, timedelta

import polars as pl

from ashare_lab.charts import write_backtest_charts


def test_nav_and_drawdown_charts_are_written(tmp_path):
    nav = pl.DataFrame(
        {
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(3)],
            "strategy_nav": [1.0, 1.1, 1.05],
            "benchmark_nav": [1.0, 1.02, 1.01],
            "excess_nav": [1.0, 1.1 / 1.02, 1.05 / 1.01],
        }
    )

    paths = write_backtest_charts(nav, tmp_path, {"enabled": True})

    assert [path.name for path in paths] == ["nav.svg", "drawdown.svg"]
    assert all(path.stat().st_size > 0 for path in paths)
