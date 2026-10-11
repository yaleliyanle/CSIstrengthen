"""TOML configuration loading.

Use dataclasses and dicts to keep fields visible without a heavyweight configuration framework.
"""

from __future__ import annotations

import copy
import difflib
import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "data": {
        "engine": "polars",
        "batch_size": 16,
        "duckdb_threads": 4,
        "market": "a share/A股行情数据.parquet",
        "valuation": "a share/A股估值数据.parquet",
        "industry": "a share/A股行业分类.parquet",
        "index_members": "a share/指数成分股.parquet",
        "factor_root": "factor",
    },
    "research": {
        "start": "2020-01-01",
        "end": "2025-12-31",
        "benchmark": "000852.SH",
        "membership_lag": 0,
    },
    "preprocess": {
        "winsor_lower": 0.01,
        "winsor_upper": 0.99,
        "standardize": "zscore",
        "neutralize_industry": False,
        "neutralize_size": False,
        "missing": "renormalize",
        "min_factor_count": 1,
    },
    "combine": {"method": "weighted"},
    "selection": {
        "method": "ic",
        "horizon": 20,
        "min_coverage": 0.8,
        "min_abs_rank_ic": 0.02,
        "min_abs_t": 2.0,
        "alpha": 0.02,
        "tolerance": 1e-7,
        "max_iter": 1000,
        "coefficient_tolerance": 1e-10,
        "top_n": 0,
        "ic_thresholds": [0.01, 0.02, 0.03],
        "alpha_grid": [0.01, 0.02, 0.05],
        "min_direction_stability": 0.6,
    },
    "multifactor": {
        "method": "max_ic_ir",
        "factor_file": "",
        "estimation_universe": "all_a",
        "lookback_periods": 12,
        "group_by_source": True,
        "exposure_shrinkage": 0.1,
        "eigenvalue_floor": 1e-8,
        "optimizer_tolerance": 1e-9,
        "optimizer_max_iterations": 1000,
        "backtest_start": "",
    },
    "split": {
        "method": "fixed",
        "train_start": "2014-10-31",
        "train_end": "2019-04-30",
        "validation_start": "2019-05-01",
        "validation_end": "2020-10-31",
        "test_start": "2020-11-01",
        "test_end": "2023-10-31",
        "live_start": "2023-11-01",
        "live_end": "2026-06-30",
    },
    "rolling": {
        "window": "expanding",
        "train_months": 24,
        "validation_months": 6,
        "step_months": 6,
    },
    "charts": {
        "enabled": True,
        "width": 1100,
        "height": 500,
    },
    "schedule": {"method": "weekly", "every_n_days": 5},
    "portfolio": {
        "method": "benchmark_tilt",
        "empty_signal": "error",
        "top_n": 100,
        "tilt": 0.35,
        "score_clip": 3.0,
        "max_weight": 0.02,
    },
    "execution": {
        "price": "next_open",
        "initial_cash": 100_000_000.0,
        "record_positions": True,
        "allow_same_close": False,
    },
    "rules": {
        "profile": "standard",
        "block_suspended": True,
        "block_limit": True,
        "use_eod_limit_for_open": False,
        "t_plus_one": True,
    },
    "costs": {
        "commission_rate": 0.0003,
        "minimum_commission": 5.0,
        "stamp_tax": "historical",
        "transfer_fee": "historical",
    },
    "analytics": {
        "horizons": [1, 5, 20],
        "quantiles": 5,
        "annual_days": 252,
        "risk_free_rate": 0.0,
        "hac_lags": "auto",
    },
    "output": {"directory": "outputs/latest"},
}

EXTRA_KEYS: dict[str, set[str]] = {
    "preprocess": {"method", "industry_column", "market_cap_column"},
    "combine": {"missing", "min_factor_count"},
    "execution": {"position_frequency", "lot_rounder"},
    "rules": {"method"},
    "costs": {"method"},
    "analytics": {"min_obs"},
}


@dataclass(frozen=True)
class FactorSpec:
    """Source and research settings for one factor."""

    name: str
    source: str | None = None
    alias: str | None = None
    direction: float = 1.0
    weight: float = 1.0
    availability_lag: int = 0

    @property
    def output_name(self) -> str:
        return self.alias or self.name

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactorSpec":
        if "name" not in value:
            raise ValueError("Every [[factors]] entry must specify name")
        item = cls(**value)
        if item.direction == 0:
            raise ValueError(f"Factor {item.name!r}: direction must not be zero")
        if item.availability_lag < 0:
            raise ValueError("availability_lag must be nonnegative")
        return item


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    research: dict[str, Any]
    factors: tuple[FactorSpec, ...]
    preprocess: dict[str, Any]
    combine: dict[str, Any]
    selection: dict[str, Any]
    multifactor: dict[str, Any]
    split: dict[str, Any]
    rolling: dict[str, Any]
    charts: dict[str, Any]
    schedule: dict[str, Any]
    portfolio: dict[str, Any]
    execution: dict[str, Any]
    rules: dict[str, Any]
    costs: dict[str, Any]
    analytics: dict[str, Any]
    output: dict[str, Any]

    @property
    def start(self) -> date:
        return date.fromisoformat(str(self.research["start"]))

    @property
    def end(self) -> date:
        return date.fromisoformat(str(self.research["end"]))

    @property
    def benchmark(self) -> str:
        return str(self.research["benchmark"])

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        _check_probable_typos(raw)
        merged = _deep_merge(DEFAULTS, raw)
        factors = tuple(FactorSpec.from_dict(item) for item in merged.pop("factors", []))
        if not factors:
            raise ValueError("Configuration requires at least one [[factors]] entry")
        config = cls(factors=factors, **merged)
        if config.start > config.end:
            raise ValueError("research.start must not be after research.end")
        if int(config.research.get("membership_lag", 0)) < 0:
            raise ValueError("research.membership_lag must be nonnegative")
        if config.data.get("engine", "polars") not in {"polars", "duckdb"}:
            raise ValueError("data.engine must be polars or duckdb")
        if int(config.data.get("batch_size", 16)) < 1:
            raise ValueError("data.batch_size must be a positive integer")
        if int(config.data.get("duckdb_threads", 4)) < 1:
            raise ValueError("data.duckdb_threads must be a positive integer")
        aliases = [factor.output_name for factor in factors]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Factor aliases must be unique")
        if config.execution["price"] not in {"same_close", "next_open"}:
            raise ValueError("execution.price must be same_close or next_open")
        if config.rules.get("profile", "standard") not in {
            "minimal",
            "standard",
            "custom",
        }:
            raise ValueError("rules.profile must be minimal, standard, or custom")
        if config.portfolio.get("empty_signal", "error") not in {
            "error",
            "hold",
            "cash",
        }:
            raise ValueError("portfolio.empty_signal must be error, hold, or cash")
        _validate_method(
            config.schedule.get("method", "weekly"),
            "schedule.method",
            {"daily", "weekly", "monthly", "quarterly", "every_n_days"},
        )
        _validate_method(
            config.combine.get("method", "weighted"),
            "combine.method",
            {"weighted", "equal", "rank_average"},
        )
        _validate_method(
            config.selection.get("method", "ic"),
            "selection.method",
            {"ic", "lasso"},
        )
        _validate_method(
            config.multifactor.get("method", "max_ic_ir"),
            "multifactor.method",
            {"equal", "max_ic_ir", "max_ic"},
        )
        if config.multifactor.get("estimation_universe", "all_a") != "all_a":
            raise ValueError("multifactor.estimation_universe currently supports only all_a")
        if int(config.multifactor.get("lookback_periods", 12)) < 2:
            raise ValueError("multifactor.lookback_periods must be at least 2")
        shrinkage = float(config.multifactor.get("exposure_shrinkage", 0.1))
        if not 0 <= shrinkage <= 1:
            raise ValueError("multifactor.exposure_shrinkage must be between 0 and 1")
        if int(config.selection.get("horizon", 20)) < 1:
            raise ValueError("selection.horizon must be a positive integer")
        if config.split.get("method", "fixed") not in {"fixed", "rolling"}:
            raise ValueError("split.method must be fixed or rolling")
        if config.rolling.get("window", "expanding") not in {
            "expanding",
            "sliding",
        }:
            raise ValueError("rolling.window must be expanding or sliding")
        _validate_method(
            config.portfolio.get("method", "benchmark_tilt"),
            "portfolio.method",
            {"benchmark_tilt", "top_n_equal", "top_n_benchmark"},
        )
        if config.preprocess.get("method"):
            _validate_method(config.preprocess["method"], "preprocess.method", set())
        if config.preprocess.get("standardize", "zscore") not in {
            "zscore",
            "rank",
            "none",
        }:
            raise ValueError("preprocess.standardize must be zscore, rank, or none")
        for section, key in (
            (config.rules, "rules.method"),
            (config.costs, "costs.method"),
            (config.execution, "execution.lot_rounder"),
            (config.charts, "charts.method"),
        ):
            leaf = key.rsplit(".", 1)[1]
            if section.get(leaf):
                _validate_method(section[leaf], key, set())
        hac_lags = config.analytics.get("hac_lags", "auto")
        if str(hac_lags).lower() != "auto" and int(hac_lags) < 0:
            raise ValueError("analytics.hac_lags must be auto or a nonnegative integer")
        return config

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["factors"] = [asdict(item) for item in self.factors]
        return value

    def effective_dict(self) -> dict[str, Any]:
        """Return resolved profile rules for persisted output and show-config."""
        value = self.to_dict()
        profile = value["rules"].get("profile", "standard")
        if profile == "minimal":
            value["rules"].update(
                block_suspended=False,
                block_limit=False,
                use_eod_limit_for_open=False,
                t_plus_one=False,
            )
            value["costs"].update(
                commission_rate=0.0,
                minimum_commission=0.0,
                stamp_tax="none",
                transfer_fee="none",
            )
            value["rules"].pop("method", None)
            value["costs"].pop("method", None)
        return value

    def to_json(self, *, effective: bool = False) -> str:
        value = self.effective_dict() if effective else self.to_dict()
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Read TOML and apply any number of ``a.b=value`` command-line overrides."""
    with Path(path).open("rb") as file:
        raw = tomllib.load(file)
    raw = apply_overrides(raw, overrides or [])
    return Config.from_dict(raw)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` to a configuration copy, including after CLI factor replacement."""
    result = copy.deepcopy(raw)
    for override in overrides:
        key, separator, text = override.partition("=")
        if not separator or not key:
            raise ValueError(f"--set must use key=value; received: {override!r}")
        _set_path(result, key.split("."), _parse_value(text))
    return result


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _check_probable_typos(raw: dict[str, Any]) -> None:
    """Allow plugin-specific parameters while catching likely misspellings of built-in keys."""
    top_level = set(DEFAULTS) | {"factors"}
    for section in raw:
        if section not in top_level:
            suggestion = difflib.get_close_matches(section, top_level, n=1)
            hint = f"; did you mean {suggestion[0]!r}" if suggestion else ""
            raise ValueError(f"Unknown configuration section {section!r}{hint}")
    for section, values in raw.items():
        if section == "factors" or not isinstance(values, dict):
            continue
        known = set(DEFAULTS.get(section, {})) | EXTRA_KEYS.get(section, set())
        for key in values:
            if key in known:
                continue
            suggestion = difflib.get_close_matches(key, known, n=1, cutoff=0.78)
            if suggestion:
                raise ValueError(
                    f"Unknown configuration key {section}.{key}; did you mean {section}.{suggestion[0]}"
                )


def _validate_method(value: Any, name: str, builtins: set[str]) -> None:
    if callable(value):
        return
    text = str(value)
    if text not in builtins and ":" not in text:
        choices = f"built-in values {sorted(builtins)} or" if builtins else ""
        raise ValueError(f"{name} must use {choices} module:function; received {text!r}")


def _parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def _set_path(root: dict[str, Any], parts: list[str], value: Any) -> None:
    current: Any = root
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current.setdefault(part, {})
    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value
