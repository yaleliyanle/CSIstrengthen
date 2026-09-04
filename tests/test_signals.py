from __future__ import annotations

from datetime import date, timedelta
import json
import math

import polars as pl
import pytest

from ashare_lab.analytics import _overlap_lags, analyze_factor, forward_returns
from ashare_lab.settings import FactorSpec
from ashare_lab.signals import (
    apply_availability_lags,
    combine_scores,
    preprocess_factors,
)


def _days(count: int) -> list[date]:
    start = date(2024, 1, 2)
    return [start + timedelta(days=index) for index in range(count)]


def test_availability_lag_is_applied_before_rebalance_filter() -> None:
    days = _days(4)
    panel = pl.DataFrame(
        {
            "date": days * 2,
            "stock": ["A"] * 4 + ["B"] * 4,
            "raw": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    spec = FactorSpec(name="raw", availability_lag=1)

    lagged = apply_availability_lags(panel, [spec])

    assert lagged.filter(pl.col("stock") == "A")["raw"].to_list() == [
        None,
        1.0,
        2.0,
        3.0,
    ]
    # 先 lag 再筛调仓日，所以得到上一交易日，而非上一调仓日。
    selected = lagged.filter(pl.col("date") == days[3]).sort("stock")
    assert selected["raw"].to_list() == [3.0, 30.0]


def test_preprocess_handles_inf_and_industry_neutralization() -> None:
    day = _days(1)[0]
    panel = pl.DataFrame(
        {
            "date": [day] * 4,
            "stock": ["A", "B", "C", "D"],
            "industry": ["I1", "I1", "I2", "I2"],
            "raw": [1.0, 3.0, 10.0, 14.0],
        }
    )
    cfg = {
        "winsor_lower": 0.0,
        "winsor_upper": 1.0,
        "neutralize_industry": True,
        "neutralize_size": False,
        "standardize": "none",
    }

    result = preprocess_factors(panel, [FactorSpec(name="raw", alias="x")], cfg)

    assert result["factor__x"].to_list() == pytest.approx([-1.0, 1.0, -2.0, 2.0])

    dirty = panel.with_columns(pl.Series("raw", [float("inf"), 0.0, 2.0, 100.0]))
    clean = preprocess_factors(
        dirty,
        [FactorSpec(name="raw")],
        {
            "winsor_lower": 0.0,
            "winsor_upper": 0.75,
            "standardize": "zscore",
        },
    )
    values = clean["factor__raw"].to_list()
    assert values[0] is None
    assert all(math.isfinite(value) for value in values[1:])
    assert sum(values[1:]) == pytest.approx(0.0)


def test_size_neutralization_removes_log_market_cap_exposure() -> None:
    logs = [1.0, 2.0, 3.0, 4.0]
    panel = pl.DataFrame(
        {
            "date": [_days(1)[0]] * 4,
            "stock": ["A", "B", "C", "D"],
            "market_cap": [math.exp(value) for value in logs],
            "raw": [2.0 + 3.0 * value for value in logs],
        }
    )
    result = preprocess_factors(
        panel,
        [FactorSpec(name="raw")],
        {
            "winsor_lower": 0.0,
            "winsor_upper": 1.0,
            "neutralize_size": True,
            "standardize": "none",
        },
    )

    assert max(abs(value) for value in result["factor__raw"]) < 1e-10


def test_size_neutralization_uses_only_rows_with_a_factor_value() -> None:
    day = _days(1)[0]
    panel = pl.DataFrame(
        {
            "date": [day] * 3,
            "stock": ["A", "B", "C"],
            "raw": [0.0, 1.0, None],
            "market_cap": [1.0, math.e, math.e**2],
        }
    )
    result = preprocess_factors(
        panel,
        [FactorSpec(name="raw")],
        {
            "winsor_lower": 0.0,
            "winsor_upper": 1.0,
            "neutralize_size": True,
            "standardize": "none",
        },
    )

    assert result["factor__raw"].to_list()[:2] == pytest.approx([0.0, 0.0])
    assert result["factor__raw"][2] is None


def test_industry_and_size_are_neutralized_jointly() -> None:
    day = _days(1)[0]
    panel = pl.DataFrame(
        {
            "date": [day] * 4,
            "stock": ["A", "B", "C", "D"],
            "industry": ["I1", "I1", "I2", "I2"],
            "market_cap": [math.exp(x) for x in [0.0, 1.0, 10.0, 11.0]],
            "raw": [0.0, 1.0, 0.0, 1.0],
        }
    )
    result = preprocess_factors(
        panel,
        [FactorSpec(name="raw")],
        {
            "winsor_lower": 0.0,
            "winsor_upper": 1.0,
            "neutralize_industry": True,
            "neutralize_size": True,
            "standardize": "none",
        },
    )

    assert max(abs(value) for value in result["factor__raw"]) < 1e-10


def test_combine_scores_keeps_rows_and_controls_missing_values() -> None:
    day = _days(1)[0]
    panel = pl.DataFrame(
        {
            "date": [day] * 3,
            "stock": ["A", "B", "C"],
            "factor__a": [1.0, None, 1.0],
            "factor__b": [None, 3.0, 3.0],
        }
    )
    specs = [FactorSpec(name="a", weight=1.0), FactorSpec(name="b", weight=3.0)]

    renormalized = combine_scores(
        panel, specs, {"method": "weighted", "missing": "renormalize"}
    )
    assert renormalized.height == panel.height
    assert renormalized["score"].to_list() == pytest.approx([1.0, 3.0, 2.5])

    require_all = combine_scores(
        panel, specs, {"method": "weighted", "missing": "require_all"}
    )
    assert require_all["score"].to_list()[:2] == [None, None]
    assert require_all["score"][2] == pytest.approx(2.5)

    minimum_two = combine_scores(
        panel,
        specs,
        {"method": "equal", "missing": "min_count", "min_factor_count": 2},
    )
    assert minimum_two["score"].to_list() == [None, None, 2.0]

    ranks = combine_scores(
        panel.fill_null(strategy="forward").fill_null(strategy="backward"),
        specs,
        {"method": "rank_average", "missing": "require_all"},
    )
    assert ranks.height == panel.height
    assert not any(column.startswith("__ashare_tmp_") for column in ranks.columns)


def test_forward_returns_use_global_calendar_not_stock_row_shift() -> None:
    days = _days(4)
    market = pl.DataFrame(
        {
            "date": days + [days[0], days[2], days[3]],
            "stock": ["A"] * 4 + ["B"] * 3,
            "open": [10.0, 12.0, 15.0, 18.0, 20.0, 30.0, 36.0],
            "close": [11.0, 13.0, 16.0, 20.0, 21.0, 31.0, 37.0],
        }
    )

    same_close = forward_returns(market, [1], "same_close")
    a0 = same_close.filter(
        (pl.col("stock") == "A") & (pl.col("date") == days[0])
    )["forward_1d"][0]
    assert a0 == pytest.approx(13.0 / 11.0 - 1.0)
    b0 = same_close.filter(
        (pl.col("stock") == "B") & (pl.col("date") == days[0])
    )["forward_1d"][0]
    assert b0 is None  # B 缺少真正的 d+1 行，不能偷用 d+2。

    next_open = forward_returns(market, [1], "next_open")
    a0_open = next_open.filter(
        (pl.col("stock") == "A") & (pl.col("date") == days[0])
    )["forward_1d"][0]
    assert a0_open == pytest.approx(15.0 / 12.0 - 1.0)
    b0_open = next_open.filter(
        (pl.col("stock") == "B") & (pl.col("date") == days[0])
    )["forward_1d"][0]
    assert b0_open is None


def test_factor_report_has_perfect_rank_ic_without_inner_join(tmp_path) -> None:
    days = _days(2)
    stocks = list("ABCDEF")
    factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2
    factors[-1] = None
    panel = pl.DataFrame(
        {
            "date": [days[0]] * 6 + [days[1]] * 6,
            "stock": stocks * 2,
            "factor__x": factors,
            "forward_1d": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] * 2,
        }
    )

    report = analyze_factor(
        panel,
        "factor__x",
        "forward_1d",
        {"quantiles": 5, "min_obs": 3},
    )

    assert report.summary["coverage"] == pytest.approx(11 / 12)
    assert report.summary["rank_ic_mean"] == pytest.approx(1.0)
    assert report.summary["rank_ic_dates"] == 2
    assert report.daily.height == 2
    assert report.quantile_returns["date"].n_unique() == 2
    assert report.summary["long_short_mean"] > 0

    paths = report.write_csv(tmp_path / "csv")
    assert all(path.exists() for path in paths.values())
    json_path = report.write_json(tmp_path / "factor.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["rank_ic_mean"] == pytest.approx(1.0)


def test_hac_lags_follow_signal_frequency_and_holding_period() -> None:
    daily = pl.Series("date", _days(30))
    weekly = pl.Series("date", _days(30)[::5])
    monthly = pl.Series("date", _days(50)[::21])
    assert _overlap_lags(20, daily, _days(30), 252, 252) == 19
    assert _overlap_lags(20, weekly, _days(30), 52, 252) == 3
    assert _overlap_lags(20, monthly, _days(50), 12, 252) == 0

    # 节后两个周频信号只隔 4 个真实交易日，5 日标签会机械重叠。
    irregular_calendar = _days(20)
    irregular = pl.Series(
        "date", [irregular_calendar[index] for index in [0, 5, 10, 14]]
    )
    assert _overlap_lags(5, irregular, irregular_calendar, 52, 252) == 1
