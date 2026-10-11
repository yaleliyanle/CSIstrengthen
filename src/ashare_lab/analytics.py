"""Forward returns and single-factor diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import polars as pl


KEYS = ["date", "stock"]


@dataclass(slots=True)
class FactorReport:
    """Summary, daily IC, and quantile-portfolio results for one factor."""

    factor: str
    forward_return: str
    summary: dict[str, Any]
    daily: pl.DataFrame
    quantile_returns: pl.DataFrame
    quantile_summary: pl.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "forward_return": self.forward_return,
            "summary": self.summary,
            "daily": self.daily.to_dicts(),
            "quantile_returns": self.quantile_returns.to_dicts(),
            "quantile_summary": self.quantile_summary.to_dicts(),
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_json_safe(self.to_dict()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def write_csv(self, directory: str | Path) -> dict[str, Path]:
        """Write four tables to a directory, using the factor name as the filename prefix."""
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(f"{self.factor}__{self.forward_return}")
        paths = {
            "summary": folder / f"{stem}__summary.csv",
            "daily": folder / f"{stem}__daily.csv",
            "quantile_returns": folder / f"{stem}__quantile_returns.csv",
            "quantile_summary": folder / f"{stem}__quantile_summary.csv",
        }
        pl.DataFrame([self.summary]).write_csv(paths["summary"])
        self.daily.write_csv(paths["daily"])
        self.quantile_returns.write_csv(paths["quantile_returns"])
        self.quantile_summary.write_csv(paths["quantile_summary"])
        return paths


def forward_returns(
    market: pl.DataFrame,
    horizons: Sequence[int],
    execution: str,
) -> pl.DataFrame:
    """Calculate returns on the market trading calendar to avoid shifts caused by missing stock rows.

    ``same_close`` measures h-day returns from close d to close d+h; ``next_open`` enters
    at open d+1 and exits at the open h trading days later. Thus next_open with h=1
    means d+1 open -> d+2 open.
    """
    if execution not in {"same_close", "next_open"}:
        raise ValueError("execution must be same_close or next_open")
    periods = [int(value) for value in horizons]
    if not periods or any(value < 1 for value in periods):
        raise ValueError("horizons must contain at least one positive integer")
    if len(periods) != len(set(periods)):
        raise ValueError("horizons must not contain duplicates")

    raw_price_col = "close" if execution == "same_close" else "open"
    adjusted_price_col = f"adj_{raw_price_col}"
    # Use adjusted prices by default; toy/custom data can supply only open/close.
    price_col = (
        adjusted_price_col if adjusted_price_col in market.columns else raw_price_col
    )
    _require_columns(market, [*KEYS, price_col])
    calendar = (
        market.select("date")
        .unique()
        .sort("date")
        .with_row_index("__day")
        .with_columns(pl.col("__day").cast(pl.Int64))
    )
    value = pl.col(price_col).cast(pl.Float64, strict=False)
    indexed = (
        market.select(*KEYS, price_col)
        .join(calendar, on="date", how="left")
        .with_columns(
            pl.when((value > 0) & value.is_finite())
            .then(value)
            .otherwise(None)
            .alias("__price")
        )
        .select(*KEYS, "__day", "__price")
    )
    result = indexed.select(*KEYS, "__day")

    start_offset = 0 if execution == "same_close" else 1
    start = indexed.select(
        "stock",
        (pl.col("__day") - start_offset).alias("__day"),
        pl.col("__price").alias("__start"),
    )
    result = result.join(start, on=["stock", "__day"], how="left")
    for horizon in periods:
        end_name = f"__end_{horizon}"
        end = indexed.select(
            "stock",
            (pl.col("__day") - start_offset - horizon).alias("__day"),
            pl.col("__price").alias(end_name),
        )
        result = result.join(end, on=["stock", "__day"], how="left").with_columns(
            (pl.col(end_name) / pl.col("__start") - 1.0).alias(
                f"forward_{horizon}d"
            )
        ).drop(end_name)
    return result.sort(KEYS).select(
        *KEYS, *[f"forward_{horizon}d" for horizon in periods]
    )


def analyze_factor(
    panel: pl.DataFrame,
    factor_col: str,
    return_col: str,
    cfg: Mapping[str, Any] | Any | None = None,
    *,
    quantiles: int | None = None,
    annual_days: int | None = None,
    min_obs: int | None = None,
    calendar: Sequence[date] | None = None,
) -> FactorReport:
    """Compute coverage, IC/RankIC, ICIR, t-statistics, quantile returns, and turnover."""
    _require_columns(panel, [*KEYS, factor_col, return_col])
    options = _analytics_options(cfg)
    groups = int(options.get("quantiles", 5) if quantiles is None else quantiles)
    annual = int(options.get("annual_days", 252) if annual_days is None else annual_days)
    minimum = int(options.get("min_obs", 3) if min_obs is None else min_obs)
    if groups < 2:
        raise ValueError("quantiles must be at least 2")
    if annual < 1 or minimum < 2:
        raise ValueError("annual_days must be positive and min_obs must be at least 2")

    raw_factor = pl.col(factor_col).cast(pl.Float64, strict=False)
    raw_return = pl.col(return_col).cast(pl.Float64, strict=False)
    frame = panel.with_columns(
        pl.when(raw_factor.is_finite())
        .then(raw_factor)
        .otherwise(None)
        .alias("__factor"),
        pl.when(raw_return.is_finite())
        .then(raw_return)
        .otherwise(None)
        .alias("__return"),
    )

    coverage = (
        frame.group_by("date")
        .agg(
            pl.len().alias("total_count"),
            pl.col("__factor").count().alias("factor_count"),
        )
        .with_columns(
            (pl.col("factor_count") / pl.col("total_count")).alias("coverage")
        )
        .sort("date")
    )
    pairs = frame.filter(
        pl.col("__factor").is_not_null() & pl.col("__return").is_not_null()
    ).with_columns(
        pl.col("__factor").rank(method="average").over("date").alias("__factor_rank"),
        pl.col("__return").rank(method="average").over("date").alias("__return_rank"),
    )
    ic = (
        pairs.group_by("date")
        .agg(
            pl.len().alias("pair_count"),
            pl.corr("__factor", "__return").alias("ic"),
            pl.corr("__factor_rank", "__return_rank").alias("rank_ic"),
        )
        .with_columns(
            pl.when((pl.col("pair_count") >= minimum) & pl.col("ic").is_finite())
            .then(pl.col("ic"))
            .otherwise(None)
            .alias("ic"),
            pl.when(
                (pl.col("pair_count") >= minimum) & pl.col("rank_ic").is_finite()
            )
            .then(pl.col("rank_ic"))
            .otherwise(None)
            .alias("rank_ic"),
        )
    )

    ranked = _quantile_members(frame, groups)
    quantile_returns = (
        ranked.group_by(["date", "quantile"])
        .agg(
            pl.col("__return").mean().alias("return"),
            pl.len().alias("member_count"),
            pl.col("__return").count().alias("observation_count"),
        )
        .sort(["date", "quantile"])
    )
    turnover = _quantile_turnover(ranked)
    quantile_returns = quantile_returns.join(
        turnover, on=["date", "quantile"], how="left"
    )
    long_short = _long_short(quantile_returns, groups)
    daily = (
        coverage.join(ic, on="date", how="left")
        .join(long_short, on="date", how="left")
        .with_columns(pl.col("pair_count").fill_null(0))
        .sort("date")
    )

    horizon = _horizon(return_col)
    signal_periods = _signal_periods_per_year(coverage["date"], annual, calendar)
    return_periods = min(signal_periods, annual / horizon)
    configured_hac = options.get("hac_lags", "auto")
    hac_lags = (
        _overlap_lags(
            horizon,
            coverage["date"],
            calendar,
            signal_periods,
            annual,
        )
        if str(configured_hac).lower() == "auto"
        else int(configured_hac)
    )
    if hac_lags < 0:
        raise ValueError("analytics.hac_lags must be nonnegative")
    quantile_summary = _summarize_quantiles(
        quantile_returns, return_periods, hac_lags
    )
    ic_stats = _series_stats(daily["ic"])
    rank_stats = _series_stats(daily["rank_ic"])
    spread_stats = _series_stats(daily["long_short"])
    ic_hac = _newey_west_stats(daily["ic"], hac_lags)
    rank_hac = _newey_west_stats(daily["rank_ic"], hac_lags)
    spread_hac = _newey_west_stats(daily["long_short"], hac_lags)
    turnover_stats = _series_stats(daily["turnover"])
    total = int(coverage["total_count"].sum() or 0)
    available = int(coverage["factor_count"].sum() or 0)
    summary = {
        "factor": factor_col,
        "forward_return": return_col,
        "rows": panel.height,
        "dates": coverage.height,
        "coverage": available / total if total else None,
        "ic_mean": ic_stats["mean"],
        "ic_std": ic_stats["std"],
        "icir_hac": _ratio(
            ic_stats["mean"], ic_hac["long_run_std"], math.sqrt(signal_periods)
        ),
        "icir_naive": _ratio(
            ic_stats["mean"], ic_stats["std"], math.sqrt(signal_periods)
        ),
        "ic_t_hac": ic_hac["t_value"],
        "ic_t_naive": ic_stats["t_value"],
        "ic_hac_lags": hac_lags,
        "ic_dates": ic_stats["count"],
        "rank_ic_mean": rank_stats["mean"],
        "rank_ic_std": rank_stats["std"],
        "rank_icir_hac": _ratio(
            rank_stats["mean"], rank_hac["long_run_std"], math.sqrt(signal_periods)
        ),
        "rank_icir_naive": _ratio(
            rank_stats["mean"], rank_stats["std"], math.sqrt(signal_periods)
        ),
        "rank_ic_t_hac": rank_hac["t_value"],
        "rank_ic_t_naive": rank_stats["t_value"],
        "rank_ic_hac_lags": hac_lags,
        "rank_ic_dates": rank_stats["count"],
        "long_short_mean": spread_stats["mean"],
        "long_short_annualized": (
            spread_stats["mean"] * return_periods
            if spread_stats["mean"] is not None
            else None
        ),
        "long_short_t_hac": spread_hac["t_value"],
        "long_short_t_naive": spread_stats["t_value"],
        "turnover_mean": turnover_stats["mean"],
    }
    return FactorReport(
        factor=factor_col,
        forward_return=return_col,
        summary=summary,
        daily=daily,
        quantile_returns=quantile_returns,
        quantile_summary=quantile_summary,
    )


def _quantile_members(frame: pl.DataFrame, groups: int) -> pl.DataFrame:
    ranked = frame.filter(pl.col("__factor").is_not_null()).with_columns(
        pl.col("__factor")
        .rank(method="average")
        .over("date")
        .cast(pl.Float64)
        .alias("__rank"),
        pl.col("__factor").count().over("date").alias("__count"),
    )
    return ranked.with_columns(
        (((pl.col("__rank") - 1.0) * groups / pl.col("__count")).floor() + 1)
        .cast(pl.Int16)
        .alias("quantile")
    )


def _quantile_turnover(ranked: pl.DataFrame) -> pl.DataFrame:
    if ranked.is_empty():
        return ranked.select("date").with_columns(
            pl.lit(None, dtype=pl.Int16).alias("quantile"),
            pl.lit(None, dtype=pl.Float64).alias("turnover"),
        ).head(0)
    calendar = (
        ranked.select("date")
        .unique()
        .sort("date")
        .with_row_index("__day")
        .with_columns(pl.col("__day").cast(pl.Int64))
    )
    weights = (
        ranked.select("date", "stock", "quantile")
        .join(calendar, on="date", how="left")
        .with_columns(
            (1.0 / pl.len().over(["date", "quantile"])).alias("__weight")
        )
        .select("__day", "stock", "quantile", "__weight")
    )
    changes = pl.concat(
        [
            weights.select(
                "__day", "stock", "quantile", pl.col("__weight").alias("__delta")
            ),
            weights.select(
                (pl.col("__day") + 1).alias("__day"),
                "stock",
                "quantile",
                (-pl.col("__weight")).alias("__delta"),
            ),
        ]
    ).filter(pl.col("__day") < calendar.height)
    turnover = (
        changes.group_by(["__day", "stock", "quantile"])
        .agg(pl.col("__delta").sum().abs().alias("__change"))
        .group_by(["__day", "quantile"])
        .agg((pl.col("__change").sum() / 2.0).alias("turnover"))
        .with_columns(
            pl.when(pl.col("__day") == 0)
            .then(None)
            .otherwise(pl.col("turnover"))
            .alias("turnover")
        )
    )
    return turnover.join(calendar, on="__day", how="left").select(
        "date", "quantile", "turnover"
    )


def _long_short(quantiles: pl.DataFrame, groups: int) -> pl.DataFrame:
    return quantiles.group_by("date").agg(
        (
            pl.col("return").filter(pl.col("quantile") == groups).first()
            - pl.col("return").filter(pl.col("quantile") == 1).first()
        ).alias("long_short"),
        (
            (
                pl.col("turnover").filter(pl.col("quantile") == groups).first()
                + pl.col("turnover").filter(pl.col("quantile") == 1).first()
            )
            / 2.0
        ).alias("turnover"),
    )


def _summarize_quantiles(
    quantile_returns: pl.DataFrame,
    periods_per_year: float,
    hac_lags: int = 0,
) -> pl.DataFrame:
    rows = []
    for group in quantile_returns.partition_by("quantile", maintain_order=True):
        if group.is_empty():
            continue
        stats = _series_stats(group["return"])
        hac = _newey_west_stats(group["return"], hac_lags)
        mean = stats["mean"]
        rows.append(
            {
                "quantile": int(group["quantile"][0]),
                "mean_return": mean,
                "annualized_return": mean * periods_per_year if mean is not None else None,
                "std_return": stats["std"],
                "t_hac": hac["t_value"],
                "t_naive": stats["t_value"],
                "periods": stats["count"],
                "mean_turnover": _series_stats(group["turnover"])["mean"],
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "quantile": pl.Int64,
            "mean_return": pl.Float64,
            "annualized_return": pl.Float64,
            "std_return": pl.Float64,
            "t_hac": pl.Float64,
            "t_naive": pl.Float64,
            "periods": pl.Int64,
            "mean_turnover": pl.Float64,
        }
    )


def _series_stats(series: pl.Series) -> dict[str, float | int | None]:
    values = np.asarray(series.drop_nulls().to_list(), dtype=float)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if not count:
        return {"count": 0, "mean": None, "std": None, "t_value": None}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else None
    t_value = (
        mean / (std / math.sqrt(count))
        if std is not None and std > 1e-14
        else None
    )
    return {"count": count, "mean": mean, "std": std, "t_value": t_value}


def _newey_west_stats(
    series: pl.Series, lags: int
) -> dict[str, float | None]:
    """Bartlett-weighted HAC accounts for dependence in overlapping forward returns."""
    values = np.asarray(series.drop_nulls().to_list(), dtype=float)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count < 2:
        return {"long_run_std": None, "t_value": None}
    lag_count = min(max(int(lags), 0), count - 1)
    centered = values - values.mean()
    long_run_variance = float(centered @ centered / count)
    for lag in range(1, lag_count + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / count)
        weight = 1.0 - lag / (lag_count + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    long_run_std = math.sqrt(long_run_variance)
    standard_error = long_run_std / math.sqrt(count)
    t_value = (
        float(values.mean() / standard_error)
        if standard_error > 1e-14
        else None
    )
    return {"long_run_std": long_run_std, "t_value": t_value}


def _ratio(value: float | None, scale: float | None, multiplier: float) -> float | None:
    return value / scale * multiplier if value is not None and scale and scale > 1e-14 else None


def _horizon(name: str) -> int:
    match = re.search(r"(?:forward_|fwd_)(\d+)d", name)
    return max(int(match.group(1)), 1) if match else 1


def _signal_periods_per_year(
    dates: pl.Series,
    annual_days: int,
    calendar: Sequence[date] | None = None,
) -> float:
    """Infer daily/weekly/monthly frequency from rebalance gaps instead of scaling monthly ICIR by sqrt(252)."""
    values = sorted(dates.drop_nulls().to_list())
    if len(values) < 2:
        return float(annual_days)
    elapsed_days = (values[-1] - values[0]).days
    if elapsed_days <= 0:
        return float(annual_days)
    observed = (len(values) - 1) * 365.25 / elapsed_days
    return min(float(annual_days), observed)


def _overlap_lags(
    horizon: int,
    dates: pl.Series,
    calendar: Sequence[date] | None,
    signal_periods: float,
    annual_days: int,
) -> int:
    """Find the maximum mechanical overlap order using complete trading-day indices."""
    values = sorted(dates.drop_nulls().to_list())
    if len(values) < 2:
        gap = annual_days / max(signal_periods, 1e-12)
        return max(math.ceil(horizon / gap) - 1, 0)
    if calendar:
        positions = {day: index for index, day in enumerate(calendar)}
        if all(day in positions for day in values):
            signal_positions = [positions[day] for day in values]
        else:
            signal_positions = []
    else:
        signal_positions = []
    if not signal_positions:
        origin = values[0].isoformat()
        signal_positions = [
            int(np.busday_count(origin, day.isoformat())) for day in values
        ]
    maximum = 0
    right = 1
    for left in range(len(signal_positions) - 1):
        right = max(right, left + 1)
        while (
            right < len(signal_positions)
            and signal_positions[right] - signal_positions[left] < horizon
        ):
            right += 1
        maximum = max(maximum, right - left - 1)
    return maximum


def _analytics_options(cfg: Mapping[str, Any] | Any | None) -> Mapping[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "analytics") and isinstance(cfg.analytics, Mapping):
        return cfg.analytics
    if isinstance(cfg, Mapping):
        nested = cfg.get("analytics")
        return nested if isinstance(nested, Mapping) else cfg
    raise TypeError("cfg must be Config, dict, or None")


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "factor"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
