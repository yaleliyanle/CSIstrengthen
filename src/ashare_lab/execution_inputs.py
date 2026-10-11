"""Execution inputs, configuration, and interface definitions.

This layer parses configuration, validates tables, and converts them to execution-engine arrays.
It contains no trading loop; start here when adding input fields or custom rules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, TypeAlias

import numpy as np
import polars as pl

from .costs import Side
from .plugins import import_callable

LotRounder: TypeAlias = Callable[[float, Side, str, date], float]
TradabilityRule: TypeAlias = Callable[["OrderContext"], str | bool | None]


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution settings; ``position_frequency`` records rebalance dates by default to save space."""

    profile: Literal["minimal", "standard", "custom"] = "standard"
    price: Literal["same_close", "next_open"] = "next_open"
    initial_cash: float = 100_000_000.0
    block_suspended: bool | None = None
    block_limit: bool | None = None
    use_eod_limit_for_open: bool | None = None
    t_plus_one: bool | None = None
    record_positions: bool = True
    position_frequency: Literal["rebalance", "daily"] = "rebalance"
    allow_target_weight_above_one: bool = False

    def __post_init__(self) -> None:
        if self.profile not in {"minimal", "standard", "custom"}:
            raise ValueError("profile must be minimal, standard, or custom")
        if self.price not in {"same_close", "next_open"}:
            raise ValueError("price must be same_close or next_open")
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.position_frequency not in {"rebalance", "daily"}:
            raise ValueError("position_frequency must be rebalance or daily")

        defaults = {
            "minimal": (False, False, False, False),
            "standard": (True, True, False, True),
            "custom": (True, True, False, True),
        }[self.profile]
        values = (
            self.block_suspended,
            self.block_limit,
            self.use_eod_limit_for_open,
            self.t_plus_one,
        )
        # minimal always disables restrictions; other profiles allow individual default overrides.
        resolved = (
            defaults
            if self.profile == "minimal"
            else tuple(
                default if item is None else bool(item)
                for item, default in zip(values, defaults)
            )
        )
        object.__setattr__(self, "block_suspended", resolved[0])
        object.__setattr__(self, "block_limit", resolved[1])
        object.__setattr__(self, "use_eod_limit_for_open", resolved[2])
        object.__setattr__(self, "t_plus_one", resolved[3])

    @classmethod
    def from_dict(
        cls,
        execution: Mapping[str, Any] | None = None,
        rules: Mapping[str, Any] | None = None,
        profile: str | None = None,
    ) -> "ExecutionConfig":
        """Accept ``Config.execution`` and ``Config.rules`` directly."""

        execution = execution or {}
        rules = rules or {}
        selected_profile = str(
            profile or execution.get("profile") or rules.get("profile") or "standard"
        )
        return cls(
            profile=selected_profile,  # type: ignore[arg-type]
            price=str(execution.get("price", "next_open")),  # type: ignore[arg-type]
            initial_cash=float(execution.get("initial_cash", 100_000_000.0)),
            record_positions=bool(execution.get("record_positions", True)),
            position_frequency=str(
                execution.get("position_frequency", "rebalance")
            ),  # type: ignore[arg-type]
            allow_target_weight_above_one=bool(
                execution.get("allow_target_weight_above_one", False)
            ),
            block_suspended=_optional_bool(rules.get("block_suspended")),
            block_limit=_optional_bool(rules.get("block_limit")),
            use_eod_limit_for_open=_optional_bool(
                rules.get("use_eod_limit_for_open")
            ),
            t_plus_one=_optional_bool(rules.get("t_plus_one")),
        )


@dataclass(frozen=True)
class OrderContext:
    """Full per-order context provided to custom tradability rules."""

    date: date
    stock: str
    side: Side
    raw_price: float
    adjusted_price: float
    trade_status: str | None
    limit_status: float | None


@dataclass(frozen=True)
class BacktestResult:
    daily_nav: pl.DataFrame
    trades: pl.DataFrame
    positions: pl.DataFrame
    rejections: pl.DataFrame


@dataclass(frozen=True)
class Target:
    signal_date: date | None
    execution_date: date
    stock: str
    weight: float


@dataclass(frozen=True)
class MarketPanel:
    calendar: list[date]
    stocks: list[str]
    stock_to_id: dict[str, int]
    starts: np.ndarray
    ends: np.ndarray
    stock_ids: np.ndarray
    raw_execution: np.ndarray
    adjusted_execution: np.ndarray
    adjusted_close: np.ndarray
    status_ok: np.ndarray
    status_codes: np.ndarray
    status_names: list[str]
    limit_status: np.ndarray


def prepare_market(
    value: pl.DataFrame | pl.LazyFrame | Sequence[Mapping[str, Any]],
    price_mode: str,
) -> MarketPanel:
    """Validate market data and convert it into compact arrays sliced by date."""

    market = _market_frame(value)
    calendar = market.get_column("date").unique().sort().to_list()
    return _make_panel(market, calendar, price_mode)


def group_targets(
    value: pl.DataFrame | pl.LazyFrame | Sequence[Mapping[str, Any]],
    calendar: list[date],
    price_mode: str,
    allow_above_one: bool = False,
) -> dict[date, list[Target]]:
    """Validate target weights and map signal dates to execution dates."""

    frame = _as_frame(value)
    required = {"stock", "target_weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"targets is missing fields: {', '.join(missing)}")
    has_execution = "execution_date" in frame.columns
    signal_columns = [name for name in ("signal_date", "date") if name in frame.columns]
    if not has_execution and len(signal_columns) != 1:
        raise ValueError("targets must provide signal_date, date, or execution_date")
    if len(signal_columns) > 1:
        raise ValueError("targets must not contain both signal_date and date")

    date_columns = (["execution_date"] if has_execution else []) + signal_columns
    frame = frame.select(*date_columns, "stock", "target_weight").with_columns(
        *(pl.col(name).cast(pl.Date, strict=True) for name in date_columns),
        pl.col("stock").cast(pl.String, strict=True),
        pl.col("target_weight").cast(pl.Float64, strict=False),
    )
    if frame.select(
        pl.any_horizontal(pl.col("stock", "target_weight").is_null())
    ).to_series().any():
        raise ValueError("targets stock/target_weight must not contain missing values")
    weights = frame.get_column("target_weight").to_numpy()
    if (~np.isfinite(weights)).any() or (weights < 0).any():
        raise ValueError("target_weight must be nonnegative and finite; execution does not support short selling")

    calendar_index = {day: index for index, day in enumerate(calendar)}
    groups: dict[date, list[Target]] = {}
    seen: set[tuple[date, str]] = set()
    for row in frame.iter_rows(named=True):
        signal_column = signal_columns[0] if signal_columns else None
        signal_day = row[signal_column] if signal_column else None
        if has_execution:
            execution_day = row["execution_date"]
            _require_trading_day(execution_day, calendar_index, "execution_date")
            if signal_day is not None:
                expected = _map_signal(signal_day, calendar, calendar_index, price_mode)
                if expected != execution_day:
                    raise ValueError(
                        f"{signal_day} with {price_mode} should execute on {expected}, "
                        f"not {execution_day}"
                    )
        else:
            execution_day = _map_signal(signal_day, calendar, calendar_index, price_mode)
        key = (execution_day, row["stock"])
        if key in seen:
            raise ValueError(
                f"Only one target portfolio per execution date is allowed; duplicate {row['stock']} on {execution_day}"
            )
        seen.add(key)
        groups.setdefault(execution_day, []).append(
            Target(signal_day, execution_day, row["stock"], float(row["target_weight"]))
        )
    for execution_day, items in groups.items():
        total = sum(item.weight for item in items)
        if not allow_above_one and total > 1.0 + 1e-10:
            raise ValueError(f"Target weights on {execution_day} total {total:.8f}, exceeding 1")
        items.sort(key=lambda item: item.stock)
    return groups


def coerce_config(
    value: ExecutionConfig | Mapping[str, Any] | Any | None,
) -> ExecutionConfig:
    """Accept standalone execution settings, project Config, or a plain dict."""

    if value is None:
        return ExecutionConfig()
    if isinstance(value, ExecutionConfig):
        return value
    if hasattr(value, "execution") and hasattr(value, "rules"):
        return ExecutionConfig.from_dict(
            value.execution,
            value.rules,
            profile=value.rules.get("profile", "standard"),
        )
    if isinstance(value, Mapping):
        if "execution" in value or "rules" in value:
            rules = value.get("rules") or {}
            return ExecutionConfig.from_dict(
                value.get("execution"),
                rules,
                profile=rules.get("profile", "standard"),
            )
        execution_keys = {
            "profile",
            "price",
            "initial_cash",
            "record_positions",
            "position_frequency",
            "allow_target_weight_above_one",
        }
        execution = {key: item for key, item in value.items() if key in execution_keys}
        rules = {key: item for key, item in value.items() if key not in execution_keys}
        return ExecutionConfig.from_dict(execution, rules)
    raise TypeError("config must be ExecutionConfig, Config, or dict")


def coerce_tradability_rule(
    value: TradabilityRule | Any | str | None,
) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = import_callable(value)
    if callable(value) or callable(getattr(value, "check", None)):
        return value
    raise TypeError("tradability_rule must implement check(), be callable, or use module:function")


def check_custom_rule(rule: Any, context: OrderContext) -> str | bool | None:
    return rule.check(context) if callable(getattr(rule, "check", None)) else rule(context)


def round_to_board_lot(
    requested_raw_shares: float,
    _side: Side,
    _stock: str,
    _day: date,
    lot: int = 100,
) -> float:
    """Optional board-lot example; execution leaves shares unrounded by default for weight research."""

    return np.floor(max(requested_raw_shares, 0.0) / lot) * lot


def keep_fractional_shares(
    requested_raw_shares: float, _side: Side, _stock: str, _day: date
) -> float:
    return requested_raw_shares


def _market_frame(
    value: pl.DataFrame | pl.LazyFrame | Sequence[Mapping[str, Any]],
) -> pl.DataFrame:
    frame = _as_frame(value)
    aliases: dict[str, str] = {}
    if "raw_open" not in frame.columns and "open" in frame.columns:
        aliases["open"] = "raw_open"
    if "raw_close" not in frame.columns and "close" in frame.columns:
        aliases["close"] = "raw_close"
    if aliases:
        frame = frame.rename(aliases)
    required = {
        "date",
        "stock",
        "raw_open",
        "raw_close",
        "adj_open",
        "adj_close",
        "trade_status",
        "limit_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"market is missing fields: {', '.join(missing)}")
    frame = frame.select(sorted(required)).with_columns(
        pl.col("date").cast(pl.Date, strict=True),
        pl.col("stock").cast(pl.String, strict=True),
        pl.col("raw_open", "raw_close", "adj_open", "adj_close").cast(
            pl.Float64, strict=False
        ),
        pl.col("limit_status").cast(pl.Float64, strict=False),
        pl.col("trade_status").cast(pl.String, strict=False),
    )
    if frame.height == 0:
        raise ValueError("market must not be empty")
    if frame.select(
        pl.any_horizontal(pl.col("date", "stock").is_null())
    ).to_series().any():
        raise ValueError("market date/stock must not contain missing values")
    duplicate = frame.select(
        pl.struct("date", "stock").is_duplicated().any().alias("duplicate")
    ).item()
    if duplicate:
        raise ValueError("market (date, stock) keys must be unique")
    return frame


def _make_panel(
    market: pl.DataFrame, calendar: list[date], price_mode: str
) -> MarketPanel:
    stocks = market.get_column("stock").unique().sort().to_list()
    stock_to_id = {stock: index for index, stock in enumerate(stocks)}
    lookup = pl.DataFrame(
        {"stock": stocks, "_stock_id": np.arange(len(stocks), dtype=np.int32)}
    )
    status_names = market.get_column("trade_status").drop_nulls().unique().sort().to_list()
    status_lookup = pl.DataFrame(
        {
            "trade_status": pl.Series(status_names, dtype=pl.String),
            "_status_code": pl.Series(
                np.arange(len(status_names), dtype=np.int16), dtype=pl.Int16
            ),
        }
    )
    # Retain the source dataset's Chinese suspension/halt values alongside English ones.
    blocked_status = ["停牌", "suspended", "suspend", "halted", "停市"]
    ordered = (
        market.join(lookup, on="stock", how="left")
        .join(status_lookup, on="trade_status", how="left")
        .with_columns(
            (~pl.col("trade_status").str.to_lowercase().is_in(blocked_status))
            .fill_null(False)
            .alias("_status_ok"),
            pl.col("_status_code").fill_null(-1).cast(pl.Int16),
        )
        .sort("date", "stock")
        .rechunk()
    )
    date_codes = ordered.get_column("date").cast(pl.Int32).to_numpy()
    changes = np.flatnonzero(date_codes[1:] != date_codes[:-1]) + 1
    starts = np.concatenate(([0], changes)).astype(np.int64)
    ends = np.concatenate((changes, [ordered.height])).astype(np.int64)
    if len(starts) != len(calendar):
        raise RuntimeError("Internal error: market date slices do not match the trading calendar")
    raw_column = "raw_close" if price_mode == "same_close" else "raw_open"
    adjusted_column = "adj_close" if price_mode == "same_close" else "adj_open"
    adjusted_close = ordered.get_column("adj_close").to_numpy()
    adjusted_execution = (
        adjusted_close
        if adjusted_column == "adj_close"
        else ordered.get_column(adjusted_column).to_numpy()
    )
    return MarketPanel(
        calendar=calendar,
        stocks=stocks,
        stock_to_id=stock_to_id,
        starts=starts,
        ends=ends,
        stock_ids=ordered.get_column("_stock_id").to_numpy(),
        raw_execution=ordered.get_column(raw_column).to_numpy(),
        adjusted_execution=adjusted_execution,
        adjusted_close=adjusted_close,
        status_ok=ordered.get_column("_status_ok").to_numpy(),
        status_codes=ordered.get_column("_status_code").to_numpy(),
        status_names=status_names,
        limit_status=ordered.get_column("limit_status").to_numpy(),
    )


def _map_signal(
    signal_day: date,
    calendar: list[date],
    calendar_index: dict[date, int],
    price_mode: str,
) -> date:
    _require_trading_day(signal_day, calendar_index, "signal_date")
    index = calendar_index[signal_day]
    if price_mode == "same_close":
        return signal_day
    if index + 1 >= len(calendar):
        raise ValueError(f"No available trading day after signal date {signal_day}")
    return calendar[index + 1]


def _require_trading_day(
    day: date, calendar_index: dict[date, int], field: str
) -> None:
    if day not in calendar_index:
        raise ValueError(f"{field}={day} is not a trading day in market")


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _as_frame(value: Any) -> pl.DataFrame:
    if isinstance(value, pl.LazyFrame):
        return value.collect()
    if isinstance(value, pl.DataFrame):
        return value
    return pl.DataFrame(value)


__all__ = [
    "BacktestResult",
    "ExecutionConfig",
    "LotRounder",
    "OrderContext",
    "TradabilityRule",
    "round_to_board_lot",
]
