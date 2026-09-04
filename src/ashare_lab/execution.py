"""A 股撮合的简洁公共入口。

日常使用通常只需要 :func:`run_execution`。如果要改某个环节：

- 费用与历史费率：``costs.py``
- 配置、输入校验、信号到成交日映射：``execution_inputs.py``
- 先卖后买、交易限制与每日记账：``execution_engine.py``

本门面继续导出旧版本的全部公共名称，因此原有导入方式无需修改。
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
    """执行目标权重，返回净值、成交、持仓和拒单四张表。

    ``targets`` 中的日期有两种口径：

    - ``signal_date`` 或 ``date``：由 ``price`` 映射；``next_open`` 严格取下一个
      全市场交易日，不会为个股停牌继续向后寻找价格。
    - ``execution_date``：已映射的成交日；若同时给出 ``signal_date``，会校验映射。

    每个日期截面被视为完整目标组合，截面未出现的旧持仓目标权重为零。
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


# 短别名方便 notebook / CLI 接入。
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
