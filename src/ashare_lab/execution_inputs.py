"""撮合输入、配置和接口定义。

本层只做三件事：解析配置、验证表格、把表格转换为撮合引擎需要的数组。
没有交易循环，适合在增加新输入字段或自定义规则时阅读。
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
    """撮合参数；``position_frequency`` 默认只记录调仓日以节省空间。"""

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
            raise ValueError("profile 只能是 minimal、standard 或 custom")
        if self.price not in {"same_close", "next_open"}:
            raise ValueError("price 只能是 same_close 或 next_open")
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash 必须是正数")
        if self.position_frequency not in {"rebalance", "daily"}:
            raise ValueError("position_frequency 只能是 rebalance 或 daily")

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
        # minimal 始终关闭规则；其他 profile 可逐项覆盖默认值。
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
        """直接接收 ``Config.execution`` 和 ``Config.rules``。"""

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
    """自定义可交易规则拿到的完整单笔上下文。"""

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
    """校验行情表，并转换为按日期切片的紧凑数组。"""

    market = _market_frame(value)
    calendar = market.get_column("date").unique().sort().to_list()
    return _make_panel(market, calendar, price_mode)


def group_targets(
    value: pl.DataFrame | pl.LazyFrame | Sequence[Mapping[str, Any]],
    calendar: list[date],
    price_mode: str,
    allow_above_one: bool = False,
) -> dict[date, list[Target]]:
    """校验目标权重，并将信号日映射为实际成交日。"""

    frame = _as_frame(value)
    required = {"stock", "target_weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"targets 缺少字段: {', '.join(missing)}")
    has_execution = "execution_date" in frame.columns
    signal_columns = [name for name in ("signal_date", "date") if name in frame.columns]
    if not has_execution and len(signal_columns) != 1:
        raise ValueError("targets 必须提供 signal_date、date 或 execution_date")
    if len(signal_columns) > 1:
        raise ValueError("targets 的 signal_date 和 date 不能同时出现")

    date_columns = (["execution_date"] if has_execution else []) + signal_columns
    frame = frame.select(*date_columns, "stock", "target_weight").with_columns(
        *(pl.col(name).cast(pl.Date, strict=True) for name in date_columns),
        pl.col("stock").cast(pl.String, strict=True),
        pl.col("target_weight").cast(pl.Float64, strict=False),
    )
    if frame.select(
        pl.any_horizontal(pl.col("stock", "target_weight").is_null())
    ).to_series().any():
        raise ValueError("targets 的 stock/target_weight 不能缺失")
    weights = frame.get_column("target_weight").to_numpy()
    if (~np.isfinite(weights)).any() or (weights < 0).any():
        raise ValueError("target_weight 必须是非负有限数；当前撮合不支持做空")

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
                        f"{signal_day} 按 {price_mode} 应在 {expected} 成交，"
                        f"而非 {execution_day}"
                    )
        else:
            execution_day = _map_signal(signal_day, calendar, calendar_index, price_mode)
        key = (execution_day, row["stock"])
        if key in seen:
            raise ValueError(
                f"同一成交日只能有一个目标组合；{execution_day} 的 {row['stock']} 重复"
            )
        seen.add(key)
        groups.setdefault(execution_day, []).append(
            Target(signal_day, execution_day, row["stock"], float(row["target_weight"]))
        )
    for execution_day, items in groups.items():
        total = sum(item.weight for item in items)
        if not allow_above_one and total > 1.0 + 1e-10:
            raise ValueError(f"{execution_day} 的目标权重合计 {total:.8f}，超过 1")
        items.sort(key=lambda item: item.stock)
    return groups


def coerce_config(
    value: ExecutionConfig | Mapping[str, Any] | Any | None,
) -> ExecutionConfig:
    """接受独立撮合配置、项目 Config 或普通 dict。"""

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
    raise TypeError("config 应为 ExecutionConfig、Config 或 dict")


def coerce_tradability_rule(
    value: TradabilityRule | Any | str | None,
) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = import_callable(value)
    if callable(value) or callable(getattr(value, "check", None)):
        return value
    raise TypeError("tradability_rule 应实现 check()、是可调用函数或 模块:函数")


def check_custom_rule(rule: Any, context: OrderContext) -> str | bool | None:
    return rule.check(context) if callable(getattr(rule, "check", None)) else rule(context)


def round_to_board_lot(
    requested_raw_shares: float,
    _side: Side,
    _stock: str,
    _day: date,
    lot: int = 100,
) -> float:
    """可选的整手示例。默认撮合不取整，便于权重研究。"""

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
        raise ValueError(f"market 缺少字段: {', '.join(missing)}")
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
        raise ValueError("market 不能为空")
    if frame.select(
        pl.any_horizontal(pl.col("date", "stock").is_null())
    ).to_series().any():
        raise ValueError("market 的 date/stock 不能缺失")
    duplicate = frame.select(
        pl.struct("date", "stock").is_duplicated().any().alias("duplicate")
    ).item()
    if duplicate:
        raise ValueError("market 的 (date, stock) 必须唯一")
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
        raise RuntimeError("内部错误：行情日期切片与交易日历不一致")
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
        raise ValueError(f"信号日 {signal_day} 之后没有可用交易日")
    return calendar[index + 1]


def _require_trading_day(
    day: date, calendar_index: dict[date, int], field: str
) -> None:
    if day not in calendar_index:
        raise ValueError(f"{field}={day} 不是 market 中的交易日")


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
