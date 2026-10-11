"""A-share execution loop.

Input validation and costs live in ``execution_inputs.py`` and ``costs.py``. This module contains
the daily valuation -> sell -> buy -> accounting flow for easy execution-logic changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from .costs import CostModel, Side, TradeCost, trade_cost
from .execution_inputs import (
    BacktestResult,
    ExecutionConfig,
    LotRounder,
    MarketPanel,
    OrderContext,
    Target,
    TradabilityRule,
    check_custom_rule,
)


@dataclass(frozen=True)
class _BuyOrder:
    target: Target
    stock_id: int
    requested_value: float


def run_engine(
    panel: MarketPanel,
    targets: dict[date, list[Target]],
    config: ExecutionConfig,
    costs: CostModel | Callable[[date, Side, float], TradeCost],
    tradability_rule: TradabilityRule | Any | None,
    lot_rounder: LotRounder,
) -> BacktestResult:
    """Run execution with validated inputs converted to arrays."""

    return _Engine(
        panel, targets, config, costs, tradability_rule, lot_rounder
    ).run()


class _Engine:
    def __init__(
        self,
        panel: MarketPanel,
        targets: dict[date, list[Target]],
        config: ExecutionConfig,
        costs: CostModel | Callable[[date, Side, float], TradeCost],
        tradability_rule: TradabilityRule | Any | None,
        lot_rounder: LotRounder,
    ) -> None:
        self.panel = panel
        self.targets = targets
        self.config = config
        self.costs = costs
        self.tradability_rule = tradability_rule
        self.lot_rounder = lot_rounder
        size = len(panel.stocks)
        self.cash = config.initial_cash
        self.shares = np.zeros(size, dtype=np.float64)  # Adjusted share quantities
        self.bought_today = np.zeros(size, dtype=np.float64)
        self.last_close = np.full(size, np.nan, dtype=np.float64)
        self.nav_rows: list[dict[str, Any]] = []
        self.trade_rows: list[dict[str, Any]] = []
        self.position_rows: list[dict[str, Any]] = []
        self.rejection_rows: list[dict[str, Any]] = []

    def run(self) -> BacktestResult:
        previous_nav = self.config.initial_cash
        for index, day in enumerate(self.panel.calendar):
            self.bought_today.fill(0.0)
            day_market = self._day_market(index)
            valuation_price = self.last_close.copy()
            valid_execution = _valid_price(day_market["adjusted"])
            valuation_price[valid_execution] = day_market["adjusted"][valid_execution]

            is_rebalance = day in self.targets
            traded_value = fees = 0.0
            execution_nav = previous_nav
            if is_rebalance:
                traded_value, fees, execution_nav = self._rebalance(
                    day, self.targets[day], valuation_price, day_market
                )

            ids = day_market["ids"]
            closes = day_market["close_rows"]
            valid_close = _valid_price(closes)
            self.last_close[ids[valid_close]] = closes[valid_close]
            # If the closing price is missing after an opening trade, value at least at the execution price.
            missing_mark = (self.shares > 0) & ~_valid_price(self.last_close)
            self.last_close[missing_mark] = day_market["adjusted"][missing_mark]

            values = np.where(
                _valid_price(self.last_close), self.shares * self.last_close, 0.0
            )
            market_value = float(values.sum())
            nav = self.cash + market_value
            daily_return = nav / previous_nav - 1.0
            self.nav_rows.append(
                {
                    "date": day,
                    "initial_nav": self.config.initial_cash,
                    "nav": nav,
                    "cash": self.cash,
                    "market_value": market_value,
                    "daily_return": daily_return,
                    "turnover": 0.5 * traded_value / max(execution_nav, 1e-12),
                    "gross_turnover": traded_value / max(execution_nav, 1e-12),
                    "fees": fees,
                    "is_rebalance": is_rebalance,
                }
            )
            if self.config.record_positions and (
                self.config.position_frequency == "daily" or is_rebalance
            ):
                self._record_positions(day, values, nav)
            previous_nav = nav

        return BacktestResult(
            daily_nav=_rows(self.nav_rows, _DAILY_SCHEMA),
            trades=_rows(self.trade_rows, _TRADE_SCHEMA),
            positions=_rows(self.position_rows, _POSITION_SCHEMA),
            rejections=_rows(self.rejection_rows, _REJECTION_SCHEMA),
        )

    def _day_market(self, index: int) -> dict[str, np.ndarray]:
        start, end = int(self.panel.starts[index]), int(self.panel.ends[index])
        ids = self.panel.stock_ids[start:end]
        size = len(self.panel.stocks)
        adjusted = np.full(size, np.nan)
        raw = np.full(size, np.nan)
        status_ok = np.zeros(size, dtype=bool)
        status_code = np.full(size, -1, dtype=np.int16)
        limit = np.full(size, np.nan)
        adjusted[ids] = self.panel.adjusted_execution[start:end]
        raw[ids] = self.panel.raw_execution[start:end]
        status_ok[ids] = self.panel.status_ok[start:end]
        status_code[ids] = self.panel.status_codes[start:end]
        limit[ids] = self.panel.limit_status[start:end]
        return {
            "ids": ids,
            "adjusted": adjusted,
            "raw": raw,
            "status_ok": status_ok,
            "status_code": status_code,
            "limit": limit,
            "close_rows": self.panel.adjusted_close[start:end],
        }

    def _rebalance(
        self,
        day: date,
        targets: list[Target],
        valuation_price: np.ndarray,
        market: dict[str, np.ndarray],
    ) -> tuple[float, float, float]:
        nav = self.cash + float(
            np.where(
                _valid_price(valuation_price),
                self.shares * valuation_price,
                0.0,
            ).sum()
        )
        by_id: dict[int, Target] = {}
        group_signal = _single_signal_date(targets)
        for target in targets:
            stock_id = self.panel.stock_to_id.get(target.stock)
            if stock_id is None:
                if target.weight > 0:
                    self._reject(
                        day,
                        target.signal_date,
                        target.stock,
                        "buy",
                        "missing_price",
                        target.weight * nav,
                    )
                continue
            by_id[stock_id] = target

        held = set(np.flatnonzero(self.shares > 1e-12).tolist())
        order_ids = sorted(held | set(by_id), key=self.panel.stocks.__getitem__)
        sells: list[tuple[int, Target, float]] = []
        buys: list[_BuyOrder] = []
        for stock_id in order_ids:
            target = by_id.get(
                stock_id,
                Target(group_signal, day, self.panel.stocks[stock_id], 0.0),
            )
            price = valuation_price[stock_id]
            current_value = (
                self.shares[stock_id] * price if _valid_scalar(price) else 0.0
            )
            difference = target.weight * nav - current_value
            tolerance = max(nav, 1.0) * 1e-12
            if difference < -tolerance:
                sells.append((stock_id, target, -difference))
            elif difference > tolerance:
                buys.append(_BuyOrder(target, stock_id, difference))

        traded_value = fees = 0.0
        for stock_id, target, requested in sells:  # Always sell before buying
            gross, cost = self._sell(day, target, stock_id, requested, market)
            traded_value += gross
            fees += cost

        allowed_buys: list[_BuyOrder] = []
        for order in buys:
            reason = self._blocked(day, order.stock_id, "buy", market)
            if reason:
                self._reject(
                    day,
                    order.target.signal_date,
                    order.target.stock,
                    "buy",
                    reason,
                    order.requested_value,
                )
            else:
                allowed_buys.append(order)

        scale = self._buy_scale(day, allowed_buys, market)
        for order in allowed_buys:
            gross, cost = self._buy(day, order, scale, market)
            traded_value += gross
            fees += cost
        return traded_value, fees, nav

    def _sell(
        self,
        day: date,
        target: Target,
        stock_id: int,
        requested: float,
        market: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        reason = self._blocked(day, stock_id, "sell", market)
        if reason:
            self._reject(day, target.signal_date, target.stock, "sell", reason, requested)
            return 0.0, 0.0

        price = float(market["adjusted"][stock_id])
        available = _sellable_shares(
            self.shares[stock_id],
            self.bought_today[stock_id],
            bool(self.config.t_plus_one),
        )
        maximum = available * price
        fill_request = min(requested, maximum)
        if fill_request <= 0:
            self._reject(
                day, target.signal_date, target.stock, "sell", "t_plus_one", requested
            )
            return 0.0, 0.0
        if fill_request < requested - 1e-9:
            self._reject(
                day,
                target.signal_date,
                target.stock,
                "sell",
                "t_plus_one",
                requested,
                requested - fill_request,
            )

        raw_shares, adjusted_shares, gross = self._rounded_fill(
            fill_request, "sell", stock_id, day, market
        )
        if gross <= 0:
            self._reject(
                day, target.signal_date, target.stock, "sell", "lot_rounding", requested
            )
            return 0.0, 0.0
        cost = trade_cost(self.costs, day, "sell", gross, target.stock)
        if cost.total >= gross:
            self._reject(
                day,
                target.signal_date,
                target.stock,
                "sell",
                "fees_exceed_proceeds",
                requested,
            )
            return 0.0, 0.0
        self.shares[stock_id] = max(0.0, self.shares[stock_id] - adjusted_shares)
        self.cash += gross - cost.total
        self._record_trade(
            day, target, "sell", raw_shares, adjusted_shares, gross, cost, market
        )
        if gross < fill_request - 1e-8:
            self._reject(
                day,
                target.signal_date,
                target.stock,
                "sell",
                "lot_rounding",
                requested,
                fill_request - gross,
            )
        return gross, cost.total

    def _buy_scale(
        self,
        day: date,
        orders: list[_BuyOrder],
        market: dict[str, np.ndarray],
    ) -> float:
        if not orders:
            return 1.0

        def needed(scale: float) -> float:
            total = 0.0
            for order in orders:
                _, _, gross = self._rounded_fill(
                    order.requested_value * scale,
                    "buy",
                    order.stock_id,
                    day,
                    market,
                )
                total += gross + trade_cost(
                    self.costs, day, "buy", gross, order.target.stock
                ).total
            return total

        if needed(1.0) <= self.cash + 1e-9:
            return 1.0
        low, high = 0.0, 1.0
        for _ in range(48):
            middle = (low + high) / 2.0
            if needed(middle) <= self.cash:
                low = middle
            else:
                high = middle
        return low

    def _buy(
        self,
        day: date,
        order: _BuyOrder,
        scale: float,
        market: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        fill_request = order.requested_value * scale
        raw_shares, adjusted_shares, gross = self._rounded_fill(
            fill_request, "buy", order.stock_id, day, market
        )
        cost = trade_cost(self.costs, day, "buy", gross, order.target.stock)
        if gross <= 0 or gross + cost.total > self.cash + 1e-8:
            self._reject(
                day,
                order.target.signal_date,
                order.target.stock,
                "buy",
                "cash_limited" if scale < 1 else "lot_rounding",
                order.requested_value,
            )
            return 0.0, 0.0

        self.cash -= gross + cost.total
        if -1e-7 < self.cash < 0:
            self.cash = 0.0
        if self.cash < 0:
            raise RuntimeError("Cash constraint failed: cash is negative after execution")
        self.shares[order.stock_id] += adjusted_shares
        self.bought_today[order.stock_id] += adjusted_shares
        self._record_trade(
            day,
            order.target,
            "buy",
            raw_shares,
            adjusted_shares,
            gross,
            cost,
            market,
        )
        if gross < order.requested_value - 1e-8:
            reason = "cash_limited" if scale < 1.0 - 1e-10 else "lot_rounding"
            self._reject(
                day,
                order.target.signal_date,
                order.target.stock,
                "buy",
                reason,
                order.requested_value,
                order.requested_value - gross,
            )
        return gross, cost.total

    def _rounded_fill(
        self,
        value: float,
        side: Side,
        stock_id: int,
        day: date,
        market: dict[str, np.ndarray],
    ) -> tuple[float, float, float]:
        if value <= 0:
            return 0.0, 0.0, 0.0
        raw_price = float(market["raw"][stock_id])
        adjusted_price = float(market["adjusted"][stock_id])
        requested_raw_shares = value / raw_price
        raw_shares = float(
            self.lot_rounder(
                requested_raw_shares, side, self.panel.stocks[stock_id], day
            )
        )
        if not np.isfinite(raw_shares) or raw_shares < 0:
            raise ValueError("lot_rounder must return a nonnegative finite share quantity")
        if raw_shares > requested_raw_shares * (1.0 + 1e-10) + 1e-9:
            raise ValueError("lot_rounder must not round an order up")
        gross = raw_shares * raw_price
        return raw_shares, gross / adjusted_price, gross

    def _blocked(
        self, day: date, stock_id: int, side: Side, market: dict[str, np.ndarray]
    ) -> str | None:
        if not (
            _valid_scalar(market["adjusted"][stock_id])
            and _valid_scalar(market["raw"][stock_id])
        ):
            return "missing_price"
        if self.config.block_suspended and not bool(market["status_ok"][stock_id]):
            return "suspended"

        limit = market["limit"][stock_id]
        # limit_status is an end-of-day field. next_open ignores execution-day closing status by default.
        use_limit = self.config.price == "same_close" or bool(
            self.config.use_eod_limit_for_open
        )
        if self.config.block_limit and use_limit and np.isfinite(limit):
            if side == "buy" and limit > 0:
                return "limit_up"
            if side == "sell" and limit < 0:
                return "limit_down"
        if self.tradability_rule is not None:
            code = int(market["status_code"][stock_id])
            status = self.panel.status_names[code] if code >= 0 else None
            context = OrderContext(
                date=day,
                stock=self.panel.stocks[stock_id],
                side=side,
                raw_price=float(market["raw"][stock_id]),
                adjusted_price=float(market["adjusted"][stock_id]),
                trade_status=status,
                limit_status=float(limit) if np.isfinite(limit) else None,
            )
            result = check_custom_rule(self.tradability_rule, context)
            if result is False:
                return "custom_rule"
            if isinstance(result, str) and result:
                return result
        return None

    def _record_trade(
        self,
        day: date,
        target: Target,
        side: Side,
        raw_shares: float,
        adjusted_shares: float,
        gross: float,
        cost: TradeCost,
        market: dict[str, np.ndarray],
    ) -> None:
        stock_id = self.panel.stock_to_id[target.stock]
        self.trade_rows.append(
            {
                "date": day,
                "signal_date": target.signal_date,
                "stock": target.stock,
                "side": side,
                "target_weight": target.weight,
                "raw_shares": raw_shares,
                "adjusted_shares": adjusted_shares,
                "raw_price": float(market["raw"][stock_id]),
                "adjusted_price": float(market["adjusted"][stock_id]),
                "gross_value": gross,
                "commission": cost.commission,
                "stamp_tax": cost.stamp_tax,
                "transfer_fee": cost.transfer_fee,
                "total_cost": cost.total,
                "cash_after": self.cash,
            }
        )

    def _reject(
        self,
        day: date,
        signal_day: date | None,
        stock: str,
        side: Side,
        reason: str,
        requested: float,
        unfilled: float | None = None,
    ) -> None:
        self.rejection_rows.append(
            {
                "date": day,
                "signal_date": signal_day,
                "stock": stock,
                "side": side,
                "reason": reason,
                "requested_value": requested,
                "unfilled_value": requested if unfilled is None else unfilled,
            }
        )

    def _record_positions(self, day: date, values: np.ndarray, nav: float) -> None:
        for stock_id in np.flatnonzero(self.shares > 1e-12):
            value = float(values[stock_id])
            self.position_rows.append(
                {
                    "date": day,
                    "stock": self.panel.stocks[stock_id],
                    "adjusted_shares": float(self.shares[stock_id]),
                    "value": value,
                    "weight": value / nav if nav else np.nan,
                }
            )


def _sellable_shares(held: float, bought_today: float, enabled: bool) -> float:
    """Single T+1 state interface: when enabled, shares bought today cannot be sold today."""

    return max(held - bought_today, 0.0) if enabled else max(held, 0.0)


def _valid_price(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def _valid_scalar(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def _single_signal_date(targets: list[Target]) -> date | None:
    days = {target.signal_date for target in targets if target.signal_date is not None}
    return next(iter(days)) if len(days) == 1 else None


def _rows(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, strict=False)


_DAILY_SCHEMA = {
    "date": pl.Date,
    "initial_nav": pl.Float64,
    "nav": pl.Float64,
    "cash": pl.Float64,
    "market_value": pl.Float64,
    "daily_return": pl.Float64,
    "turnover": pl.Float64,
    "gross_turnover": pl.Float64,
    "fees": pl.Float64,
    "is_rebalance": pl.Boolean,
}

_TRADE_SCHEMA = {
    "date": pl.Date,
    "signal_date": pl.Date,
    "stock": pl.String,
    "side": pl.String,
    "target_weight": pl.Float64,
    "raw_shares": pl.Float64,
    "adjusted_shares": pl.Float64,
    "raw_price": pl.Float64,
    "adjusted_price": pl.Float64,
    "gross_value": pl.Float64,
    "commission": pl.Float64,
    "stamp_tax": pl.Float64,
    "transfer_fee": pl.Float64,
    "total_cost": pl.Float64,
    "cash_after": pl.Float64,
}

_POSITION_SCHEMA = {
    "date": pl.Date,
    "stock": pl.String,
    "adjusted_shares": pl.Float64,
    "value": pl.Float64,
    "weight": pl.Float64,
}

_REJECTION_SCHEMA = {
    "date": pl.Date,
    "signal_date": pl.Date,
    "stock": pl.String,
    "side": pl.String,
    "reason": pl.String,
    "requested_value": pl.Float64,
    "unfilled_value": pl.Float64,
}
