from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from ashare_lab.pipeline import (
    _apply_empty_signal_policy,
    build_research_panel,
    run_backtest,
)
from ashare_lab.settings import Config


def _weekdays(start: date, count: int) -> list[date]:
    days = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _toy_project(tmp_path: Path) -> dict[str, str]:
    data_root = tmp_path / "a share"
    factor_root = tmp_path / "factor" / "toy"
    data_root.mkdir(parents=True)
    factor_root.mkdir(parents=True)
    days = _weekdays(date(2024, 1, 2), 27)
    stocks = ["A.SZ", "B.SH", "C.BJ"]

    market_rows = []
    valuation_rows = []
    factor_rows = []
    prices = {"A.SZ": 10.0, "B.SH": 10.0, "C.BJ": 10.0}
    growth = {"A.SZ": 1.01, "B.SH": 1.0, "C.BJ": 0.99}
    alpha = {"A.SZ": 3.0, "B.SH": 2.0, "C.BJ": 1.0}
    for day in days:
        for stock in stocks:
            opened = prices[stock]
            closed = opened * growth[stock]
            market_rows.append(
                {
                    "s_info_windcode": stock,
                    "trade_dt": day,
                    "s_dq_open": opened,
                    "s_dq_close": closed,
                    "s_dq_adjopen": opened,
                    "s_dq_adjclose": closed,
                    "s_dq_tradestatus": "交易",
                }
            )
            valuation_rows.append(
                {
                    "s_info_windcode": stock,
                    "trade_dt": day,
                    "s_val_mv": 100.0,
                    # Standard rules reject the top-scoring stock; minimal rules still permit the trade.
                    "up_down_limit_status": 1 if stock == "A.SZ" else 0,
                }
            )
            factor_rows.append(
                {
                    "trade_dt": day,
                    "s_info_windcode": stock,
                    "alpha": alpha[stock],
                }
            )
            prices[stock] = closed

    pl.DataFrame(market_rows).write_parquet(data_root / "A股行情数据.parquet")
    pl.DataFrame(valuation_rows).write_parquet(data_root / "A股估值数据.parquet")
    pl.DataFrame(
        {
            "s_info_windcode": ["IDX"] * 3,
            "s_con_windcode": stocks,
            "trade_dt": [days[0]] * 3,
            "i_weight": [1 / 3] * 3,
        }
    ).write_parquet(data_root / "指数成分股.parquet")
    pl.DataFrame(
        {
            **{stock: ["industry"] * len(days) for stock in stocks},
            "trade_dt": days,
        }
    ).write_parquet(data_root / "A股行业分类.parquet")
    pl.DataFrame(factor_rows).write_parquet(factor_root / "daily_toy_2024.parquet")
    return {
        "market": str(data_root / "A股行情数据.parquet"),
        "valuation": str(data_root / "A股估值数据.parquet"),
        "industry": str(data_root / "A股行业分类.parquet"),
        "index_members": str(data_root / "指数成分股.parquet"),
        "factor_root": str(tmp_path / "factor"),
    }


def _config(tmp_path: Path, *, rebalance: str, rules: str) -> Config:
    paths = _toy_project(tmp_path)
    return Config.from_dict(
        {
            "data": paths,
            "research": {
                "start": "2024-01-02",
                "end": "2024-01-31",
                "benchmark": "IDX",
            },
            "factors": [{"source": "toy", "name": "alpha"}],
            "preprocess": {
                "neutralize_industry": False,
                "neutralize_size": False,
            },
            "schedule": {"method": rebalance},
            "portfolio": {"method": "top_n_equal", "top_n": 1},
            "execution": {"price": "same_close", "initial_cash": 1_000_000.0},
            "rules": {"profile": rules},
            "output": {"directory": str(tmp_path / "outputs")},
        }
    )


def test_weekly_and_monthly_schedules_work_end_to_end(tmp_path):
    weekly = _config(tmp_path / "weekly", rebalance="weekly", rules="minimal")
    monthly = _config(tmp_path / "monthly", rebalance="monthly", rules="minimal")

    weekly_panel = build_research_panel(weekly)
    monthly_panel = build_research_panel(monthly)

    assert len(weekly_panel.signal_dates) > len(monthly_panel.signal_dates) == 1
    result = run_backtest(weekly)
    assert result.summary["run"]["rebalance"] == "weekly"
    assert result.summary["run"]["rules_profile"] == "minimal"
    assert result.strategy.trades.height > 0
    # Reset index weights only when snapshots change, not on every weekly strategy rebalance.
    assert result.benchmark.trades.height == 3
    assert (result.output_directory / "summary.json").exists()


def test_partial_week_end_is_not_mistaken_for_week_end(tmp_path):
    config = _config(tmp_path, rebalance="weekly", rules="minimal")
    raw = config.to_dict()
    raw["research"]["end"] = "2024-01-24"  # Wednesday
    panel = build_research_panel(Config.from_dict(raw))

    assert panel.signal_dates[-1] == date(2024, 1, 19)
    assert date(2024, 1, 24) not in panel.signal_dates


def test_duckdb_backend_matches_polars_end_to_end(tmp_path):
    polars_config = _config(tmp_path, rebalance="weekly", rules="minimal")
    raw = polars_config.to_dict()
    raw["data"]["engine"] = "duckdb"
    raw["data"]["batch_size"] = 1
    raw["output"]["directory"] = str(tmp_path / "duckdb_outputs")
    duckdb_config = Config.from_dict(raw)

    polars_result = run_backtest(polars_config)
    duckdb_result = run_backtest(duckdb_config)

    assert duckdb_result.comparison.to_dicts() == polars_result.comparison.to_dicts()
    assert duckdb_result.targets.to_dicts() == polars_result.targets.to_dicts()


def test_empty_signal_policy_is_explicit():
    first, second = date(2024, 1, 5), date(2024, 1, 12)
    targets = pl.DataFrame(
        {"date": [first], "stock": ["A"], "target_weight": [1.0]}
    )
    panel = pl.DataFrame(
        {"date": [first, second], "stock": ["A", "B"], "score": [1.0, None]}
    )

    with pytest.raises(ValueError, match="No valid scores"):
        _apply_empty_signal_policy(targets, panel, [first, second], "error")
    held = _apply_empty_signal_policy(targets, panel, [first, second], "hold")
    cash = _apply_empty_signal_policy(targets, panel, [first, second], "cash")
    assert held["date"].to_list() == [first]
    assert cash.filter(pl.col("date") == second).item(0, "target_weight") == 0.0


def test_minimal_profile_really_removes_limit_and_cost_rules(tmp_path):
    minimal = run_backtest(
        _config(tmp_path / "minimal", rebalance="monthly", rules="minimal")
    )
    standard = run_backtest(
        _config(tmp_path / "standard", rebalance="monthly", rules="standard")
    )

    assert minimal.strategy.trades.height == 1
    assert minimal.strategy.daily_nav["fees"].sum() == 0
    assert standard.strategy.trades.is_empty()
    assert "limit_up" in standard.strategy.rejections["reason"].to_list()
