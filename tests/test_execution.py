from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ashare_lab.execution import (
    CostContext,
    ExecutionConfig,
    HistoricalAshareCosts,
    _sellable_shares,
    historical_stamp_tax_rate,
    historical_transfer_fee_rate,
    run_execution,
)
from ashare_lab.metrics import performance_metrics


ZERO_COST = HistoricalAshareCosts(
    commission_rate=0,
    minimum_commission=0,
    stamp_tax=False,
    transfer_fee=False,
)


def _row(
    day: date,
    stock: str,
    *,
    open_: float | None = 10.0,
    close: float | None = 10.0,
    status: str = "交易",
    limit: int | None = 0,
) -> dict[str, object]:
    return {
        "date": day,
        "stock": stock,
        "raw_open": open_,
        "raw_close": close,
        "adj_open": open_,
        "adj_close": close,
        "trade_status": status,
        "limit_status": limit,
    }


def _config(
    *,
    price: str = "same_close",
    cash: float = 10_000.0,
    profile: str = "standard",
) -> ExecutionConfig:
    return ExecutionConfig(profile=profile, price=price, initial_cash=cash)


def test_same_close_and_next_open_are_different_and_skip_weekend() -> None:
    friday = date(2023, 8, 25)
    monday = date(2023, 8, 28)
    tuesday = date(2023, 8, 29)
    market = pl.DataFrame(
        [
            _row(friday, "A", open_=100, close=110),
            _row(monday, "A", open_=120, close=132),
            _row(tuesday, "A", open_=132, close=132),
        ]
    )
    targets = pl.DataFrame(
        {"signal_date": [friday], "stock": ["A"], "target_weight": [1.0]}
    )

    at_close = run_execution(
        market,
        targets,
        config=_config(price="same_close"),
        cost_model=ZERO_COST,
    )
    next_open = run_execution(
        market,
        targets,
        config=_config(price="next_open"),
        cost_model=ZERO_COST,
    )

    assert at_close.trades.row(0, named=True)["date"] == friday
    assert at_close.trades.row(0, named=True)["adjusted_price"] == 110
    assert next_open.trades.row(0, named=True)["date"] == monday
    assert next_open.trades.row(0, named=True)["adjusted_price"] == 120
    assert at_close.daily_nav.filter(pl.col("date") == monday).item(0, "nav") == pytest.approx(
        12_000
    )
    assert next_open.daily_nav.filter(pl.col("date") == monday).item(0, "nav") == pytest.approx(
        11_000
    )

    weekend_signal = targets.with_columns(pl.lit(date(2023, 8, 26)).alias("signal_date"))
    with pytest.raises(ValueError, match="不是 market 中的交易日"):
        run_execution(market, weekend_signal, config=_config(price="next_open"))


def test_suspension_missing_prices_and_limits_block_both_sides() -> None:
    first = date(2024, 1, 2)
    second = date(2024, 1, 3)
    stocks = ["A", "B", "C", "D", "E", "F"]
    rows = [_row(first, stock) for stock in stocks]
    rows += [
        _row(second, "A", status="停牌"),
        _row(second, "B", limit=-1),
        _row(second, "C"),
        _row(second, "D", limit=1),
        _row(second, "E", close=None),
        _row(second, "F", close=None),
    ]
    market = pl.DataFrame(rows)
    targets = pl.DataFrame(
        {
            "date": [first] * 4 + [second] * 2,
            "stock": ["A", "B", "C", "F", "D", "E"],
            "target_weight": [0.2, 0.2, 0.2, 0.2, 0.5, 0.5],
        }
    )

    result = run_execution(
        market, targets, config=_config(), cost_model=ZERO_COST
    )
    rejected = {
        row["stock"]: row["reason"]
        for row in result.rejections.filter(pl.col("date") == second).iter_rows(named=True)
    }
    assert rejected == {
        "A": "suspended",
        "B": "limit_down",
        "D": "limit_up",
        "E": "missing_price",
        "F": "missing_price",
    }
    assert result.trades.filter(
        (pl.col("date") == second) & (pl.col("stock") == "C") & (pl.col("side") == "sell")
    ).height == 1
    held = set(
        result.positions.filter(pl.col("date") == second)
        .get_column("stock")
        .to_list()
    )
    assert held == {"A", "B", "F"}  # 三笔拒单的旧持仓原样保留


@pytest.mark.parametrize("status", ["XD", "XR", "DR", "N"])
def test_special_trade_statuses_are_tradable(status: str) -> None:
    day = date(2024, 2, 1)
    result = run_execution(
        pl.DataFrame([_row(day, "A", status=status)]),
        pl.DataFrame({"execution_date": [day], "stock": ["A"], "target_weight": [1.0]}),
        config=_config(),
        cost_model=ZERO_COST,
    )
    assert result.trades.get_column("side").to_list() == ["buy"]
    assert result.rejections.is_empty()


@pytest.mark.parametrize(
    ("day", "buy_rate", "sell_rate"),
    [
        (date(2005, 1, 23), 0.002, 0.002),
        (date(2005, 1, 24), 0.001, 0.001),
        (date(2007, 5, 29), 0.001, 0.001),
        (date(2007, 5, 30), 0.003, 0.003),
        (date(2008, 4, 23), 0.003, 0.003),
        (date(2008, 4, 24), 0.001, 0.001),
        (date(2008, 9, 18), 0.001, 0.001),
        (date(2008, 9, 19), 0.0, 0.001),
        (date(2023, 8, 27), 0.0, 0.001),
        (date(2023, 8, 28), 0.0, 0.0005),
    ],
)
def test_stamp_tax_boundaries(day: date, buy_rate: float, sell_rate: float) -> None:
    assert historical_stamp_tax_rate(day, "buy") == buy_rate
    assert historical_stamp_tax_rate(day, "sell") == sell_rate


def test_transfer_fee_commission_and_fixed_rate_overrides() -> None:
    assert historical_transfer_fee_rate(date(2022, 4, 28), "buy") == 0.00002
    assert historical_transfer_fee_rate(date(2022, 4, 29), "sell") == 0.00001

    minimum = HistoricalAshareCosts(stamp_tax=False, transfer_fee=False)
    assert minimum.calculate(date(2024, 1, 1), "buy", 1_000).commission == 5
    assert minimum.calculate(date(2024, 1, 1), "buy", 100_000).commission == pytest.approx(30)

    fixed = HistoricalAshareCosts(
        commission_rate=0,
        minimum_commission=0,
        stamp_tax=0.0005,
        transfer_fee=0.0001,
    )
    cost = fixed.calculate(date(2024, 1, 1), "buy", 100_000)
    assert cost.stamp_tax == 50
    assert cost.transfer_fee == 10


def test_cash_constraint_and_accounting_conserve_cash() -> None:
    day = date(2024, 1, 2)
    costs = HistoricalAshareCosts(
        commission_rate=0.001,
        minimum_commission=5,
        stamp_tax=False,
        transfer_fee=False,
    )
    result = run_execution(
        pl.DataFrame([_row(day, "A", open_=1, close=1)]),
        pl.DataFrame({"date": [day], "stock": ["A"], "target_weight": [1.0]}),
        config=_config(cash=100),
        cost_model=costs,
    )
    trade = result.trades.row(0, named=True)
    assert trade["gross_value"] == pytest.approx(95.0, abs=1e-8)
    assert trade["commission"] == 5.0
    assert trade["cash_after"] == pytest.approx(0.0, abs=1e-8)
    assert trade["gross_value"] + trade["total_cost"] + trade["cash_after"] == pytest.approx(
        100.0
    )
    nav = result.daily_nav.row(0, named=True)
    assert nav["nav"] == pytest.approx(nav["cash"] + nav["market_value"])
    assert nav["nav"] == pytest.approx(95.0)
    assert nav["initial_nav"] == 100.0
    assert performance_metrics(result.daily_nav)["total_return"] == pytest.approx(-0.05)
    assert result.rejections.get_column("reason").to_list() == ["cash_limited"]


def test_raw_weights_above_one_can_be_kept_without_negative_cash() -> None:
    day = date(2024, 1, 2)
    market = pl.DataFrame([_row(day, "A"), _row(day, "B")])
    targets = pl.DataFrame(
        {
            "date": [day, day],
            "stock": ["A", "B"],
            "target_weight": [0.5001, 0.5001],
        }
    )

    with pytest.raises(ValueError, match="超过 1"):
        run_execution(market, targets, config=_config(), cost_model=ZERO_COST)

    result = run_execution(
        market,
        targets,
        config=ExecutionConfig(
            profile="custom",
            price="same_close",
            initial_cash=10_000,
            allow_target_weight_above_one=True,
        ),
        cost_model=ZERO_COST,
    )

    assert result.trades.get_column("target_weight").to_list() == [0.5001, 0.5001]
    assert result.daily_nav.item(0, "cash") >= 0
    assert result.rejections.get_column("reason").to_list() == [
        "cash_limited",
        "cash_limited",
    ]


def test_next_open_does_not_use_execution_day_eod_limit_by_default() -> None:
    friday, monday = date(2024, 2, 2), date(2024, 2, 5)
    market = pl.DataFrame(
        [_row(friday, "A"), _row(monday, "A", open_=9.8, close=10.8, limit=1)]
    )
    targets = pl.DataFrame(
        {"date": [friday], "stock": ["A"], "target_weight": [1.0]}
    )

    normal = run_execution(
        market,
        targets,
        config=ExecutionConfig(
            profile="standard", price="next_open", initial_cash=10_000
        ),
        cost_model=ZERO_COST,
    )
    conservative = run_execution(
        market,
        targets,
        config=ExecutionConfig(
            profile="custom",
            price="next_open",
            initial_cash=10_000,
            use_eod_limit_for_open=True,
        ),
        cost_model=ZERO_COST,
    )

    assert normal.trades.item(0, "date") == monday
    assert normal.trades.item(0, "adjusted_price") == pytest.approx(9.8)
    assert conservative.trades.is_empty()
    assert conservative.rejections.item(0, "reason") == "limit_up"


def test_t_plus_one_locks_today_buys_but_releases_next_day() -> None:
    assert _sellable_shares(100, 100, True) == 0
    assert _sellable_shares(100, 100, False) == 100
    assert _sellable_shares(100, 0, True) == 100

    first, second = date(2024, 1, 2), date(2024, 1, 3)
    market = pl.DataFrame(
        [_row(first, "A"), _row(first, "B"), _row(second, "A"), _row(second, "B")]
    )
    targets = pl.DataFrame(
        {
            "date": [first, second],
            "stock": ["A", "B"],
            "target_weight": [1.0, 0.0],
        }
    )
    result = run_execution(market, targets, config=_config(), cost_model=ZERO_COST)
    assert result.trades.select("date", "stock", "side").rows() == [
        (first, "A", "buy"),
        (second, "A", "sell"),
    ]

    duplicate = pl.DataFrame(
        {
            "date": [first, first],
            "stock": ["A", "A"],
            "target_weight": [0.5, 0.5],
        }
    )
    with pytest.raises(ValueError, match="同一成交日只能有一个目标组合"):
        run_execution(market, duplicate, config=_config(), cost_model=ZERO_COST)


def test_minimal_standard_and_custom_profiles_are_explicit() -> None:
    day = date(2024, 1, 2)
    market = pl.DataFrame(
        [
            _row(day, "HALT", status="停牌"),
            _row(day, "UP", limit=1),
        ]
    ).rename({"raw_open": "open", "raw_close": "close"})  # 两套原价字段均兼容
    targets = pl.DataFrame(
        {"date": [day, day], "stock": ["HALT", "UP"], "target_weight": [0.5, 0.5]}
    )
    expensive = HistoricalAshareCosts(
        commission_rate=0.01, stamp_tax=0.01, transfer_fee=0.01
    )

    minimal = run_execution(
        market,
        targets,
        config=_config(profile="minimal"),
        cost_model=expensive,  # minimal 明确忽略所有费用
    )
    standard = run_execution(
        market,
        targets,
        config=_config(profile="standard"),
        cost_model=ZERO_COST,
    )
    assert minimal.trades.height == 2
    assert minimal.daily_nav.item(0, "fees") == 0
    assert standard.trades.is_empty()
    assert set(standard.rejections.get_column("reason")) == {"suspended", "limit_up"}

    from_dict = ExecutionConfig.from_dict(
        {"profile": "minimal", "price": "same_close"},
        {"block_suspended": True, "block_limit": True, "t_plus_one": True},
    )
    assert (from_dict.block_suspended, from_dict.block_limit, from_dict.t_plus_one) == (
        False,
        False,
        False,
    )
    custom = ExecutionConfig.from_dict(
        {"price": "same_close"},
        {"block_suspended": False, "block_limit": True, "t_plus_one": False},
        profile="custom",
    )
    assert (custom.block_suspended, custom.block_limit, custom.t_plus_one) == (
        False,
        True,
        False,
    )


def test_custom_rules_and_module_function_costs_are_pluggable() -> None:
    day = date(2024, 1, 2)
    market = pl.DataFrame([_row(day, "IPO", status="N")])
    targets = pl.DataFrame(
        {"date": [day], "stock": ["IPO"], "target_weight": [1.0]}
    )

    rejected = run_execution(
        market,
        targets,
        config=_config(profile="custom"),
        cost_model="ashare_lab.execution:no_costs",
        tradability_rule=lambda context: "custom_new_stock" if context.trade_status == "N" else None,
    )
    assert rejected.trades.is_empty()
    assert rejected.rejections.item(0, "reason") == "custom_new_stock"


def test_cost_plugin_can_receive_stock_context() -> None:
    day = date(2024, 1, 2)
    seen: list[str] = []

    def stock_cost(context: CostContext):
        seen.append(context.stock)
        return ZERO_COST.calculate(context.date, context.side, context.value)

    result = run_execution(
        pl.DataFrame([_row(day, "600000.SH")]),
        pl.DataFrame(
            {"date": [day], "stock": ["600000.SH"], "target_weight": [1.0]}
        ),
        config=_config(),
        cost_model=stock_cost,
    )
    assert result.trades.height == 1
    assert seen and set(seen) == {"600000.SH"}
