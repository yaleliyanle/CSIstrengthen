from datetime import date, timedelta

import polars as pl
import pytest

from ashare_lab.metrics import compare_with_benchmark, performance_metrics
from ashare_lab.portfolio import build_targets
from ashare_lab.schedule import make_schedule
from ashare_lab.settings import Config


def _config(**sections):
    raw = {"factors": [{"name": "quality"}]}
    raw.update(sections)
    return Config.from_dict(raw)


def test_config_defaults_and_factor_alias_are_explicit():
    config = _config(factors=[{"name": "roe", "alias": "quality", "direction": -1}])
    assert config.execution["price"] == "next_open"
    assert config.factors[0].output_name == "quality"
    assert config.factors[0].direction == -1


def test_config_catches_likely_command_line_typos():
    with pytest.raises(ValueError, match="schedule.method"):
        _config(schedule={"metod": "monthly"})
    with pytest.raises(ValueError, match="portfolio.method"):
        _config(portfolio={"method": "not_a_method"})


def test_minimal_effective_config_removes_ignored_plugins():
    config = _config(
        rules={"profile": "minimal", "method": "x:y"},
        costs={"method": "x:z"},
    )
    effective = config.effective_dict()
    assert "method" not in effective["rules"]
    assert "method" not in effective["costs"]


def test_weekly_schedule_uses_last_real_trading_day():
    monday = date(2024, 1, 1)
    calendar = [monday + timedelta(days=i) for i in [0, 1, 2, 3, 4, 7, 8]]
    assert make_schedule(calendar, {"method": "weekly"}) == [
        date(2024, 1, 5),
        date(2024, 1, 9),
    ]


def test_benchmark_tilt_is_normalized_and_capped():
    day = date(2024, 1, 2)
    panel = pl.DataFrame(
        {
            "date": [day] * 5,
            "stock": list("ABCDE"),
            "score": [10.0, 1.0, 0.0, -1.0, -2.0],
            "benchmark_weight": [0.2] * 5,
        }
    )
    result = build_targets(
        panel,
        {
            "method": "benchmark_tilt",
            "tilt": 1.0,
            "score_clip": 3.0,
            "max_weight": 0.3,
        },
    )
    assert result["target_weight"].sum() == pytest.approx(1.0)
    assert result["target_weight"].max() <= 0.3 + 1e-12


def test_performance_metrics_detect_drawdown():
    daily = pl.DataFrame(
        {
            "date": [date(2024, 1, i) for i in range(1, 5)],
            "nav": [100.0, 110.0, 99.0, 120.0],
            "daily_return": [0.0, 0.1, -0.1, 120 / 99 - 1],
        }
    )
    result = performance_metrics(daily)
    assert result["total_return"] == pytest.approx(0.2)
    assert result["max_drawdown"] == pytest.approx(-0.1)


def test_relative_nav_is_strategy_wealth_divided_by_benchmark_wealth():
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    strategy = pl.DataFrame(
        {
            "date": days,
            "initial_nav": [100.0, 100.0],
            "nav": [110.0, 121.0],
            "daily_return": [0.1, 0.1],
        }
    )
    benchmark = pl.DataFrame(
        {
            "date": days,
            "initial_nav": [100.0, 100.0],
            "nav": [105.0, 110.25],
            "daily_return": [0.05, 0.05],
        }
    )
    comparison, summary = compare_with_benchmark(
        strategy, benchmark, risk_free_rate=0.02
    )
    expected = 1.21 / 1.1025
    assert comparison["excess_nav"][-1] == pytest.approx(expected)
    assert summary["active"]["total_excess_return"] == pytest.approx(expected - 1)
