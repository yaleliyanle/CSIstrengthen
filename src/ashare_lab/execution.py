"""Concise public entry point for A-share execution.

Most users only need :func:`run_execution`. To modify individual components:

- Costs and historical rates: ``costs.py``
- Configuration, input validation, and signal-to-execution date mapping: ``execution_inputs.py``
- Sell-before-buy execution, trading restrictions, and daily accounting: ``execution_engine.py``

This facade still exports all public names from earlier versions, preserving existing imports.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

import polars as pl

from .costs import (
    CostContext,
    CostModel,
    FeeRule,
    HistoricalAshareCosts,
    NoCosts,
    Side,
    TradeCost,
    coerce_cost_model,
    historical_stamp_tax,
    historical_stamp_tax_rate,
    historical_transfer_fee,
    historical_transfer_fee_rate,
    no_costs,
)
from .execution_engine import _sellable_shares, run_engine
from .execution_inputs import (
    BacktestResult,
    ExecutionConfig,
    LotRounder,
    OrderContext,
    TradabilityRule,
    coerce_config,
    coerce_tradability_rule,
    group_targets,
    keep_fractional_shares,
    prepare_market,
    round_to_board_lot,
)


def run_execution(
    market: pl.DataFrame | pl.LazyFrame | Sequence[Mapping[str, Any]],
    targets: pl.DataFrame | pl.LazyFrame | Sequence[Mapping[str, Any]],
    *,
    config: ExecutionConfig | Mapping[str, Any] | Any | None = None,
    cost_model: CostModel
    | Callable[[date, Side, float], TradeCost]
    | Mapping[str, Any]
    | str
    | None = None,
    tradability_rule: TradabilityRule | Any | str | None = None,
    lot_rounder: LotRounder | None = None,
) -> BacktestResult:
    """Execute target weights and return NAV, trades, positions, and rejections.

    ``targets`` supports two date conventions:

    - ``signal_date`` or ``date``: map using ``price``; ``next_open`` always uses the next
      market trading day without searching further ahead for suspended stocks.
    - ``execution_date``: already mapped; supplying ``signal_date`` also validates the mapping.

    Each date is a complete target portfolio; existing positions omitted from it receive zero target weight.
    """

    raw_config = config
    resolved = coerce_config(config)
    if cost_model is None and hasattr(raw_config, "costs"):
        cost_model = raw_config.costs

    costs = (
        NoCosts()
        if resolved.profile == "minimal"
        else coerce_cost_model(cost_model)
    )
    trade_rule = (
        None
        if resolved.profile == "minimal"
        else coerce_tradability_rule(tradability_rule)
    )
    panel = prepare_market(market, resolved.price)
    target_groups = group_targets(
        targets,
        panel.calendar,
        resolved.price,
        allow_above_one=resolved.allow_target_weight_above_one,
    )
    return run_engine(
        panel,
        target_groups,
        resolved,
        costs,
        trade_rule,
        lot_rounder or keep_fractional_shares,
    )


# Short aliases for notebook and CLI use.
backtest = run_execution


__all__ = [
    "BacktestResult",
    "CostModel",
    "CostContext",
    "ExecutionConfig",
    "FeeRule",
    "HistoricalAshareCosts",
    "LotRounder",
    "NoCosts",
    "OrderContext",
    "Side",
    "TradeCost",
    "TradabilityRule",
    "backtest",
    "historical_stamp_tax",
    "historical_stamp_tax_rate",
    "historical_transfer_fee",
    "historical_transfer_fee_rate",
    "no_costs",
    "round_to_board_lot",
    "run_execution",
]
