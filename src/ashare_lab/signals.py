"""Factor lags, cross-sectional preprocessing, and combination.

Keys are ``date`` / ``stock`` and processed factors are named
``factor__<alias>``. Configure custom combination methods as
``your_module:your_function``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from .plugins import import_callable
from .settings import FactorSpec


KEYS = ["date", "stock"]


def apply_availability_lags(
    daily_factors: pl.DataFrame, specs: Sequence[FactorSpec]
) -> pl.DataFrame:
    """Delay raw factors by their availability lag for each stock.

    One day corresponds to one input row per stock. Apply this function to a **complete,
    continuous daily factor table** before selecting weekly/monthly rebalance dates; otherwise lag=1
    would incorrectly shift by an entire rebalance period.

    Overwrite raw factor columns while preserving the row count.
    """
    _require_columns(daily_factors, KEYS)
    names = [_spec_value(spec, "name") for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("apply_availability_lags does not support duplicate factor names")
    _require_columns(daily_factors, names)

    frame = daily_factors.sort(KEYS[::-1])  # stock, then date
    expressions = []
    for spec in specs:
        name = str(_spec_value(spec, "name"))
        lag = int(_spec_value(spec, "availability_lag", 0))
        if lag < 0:
            raise ValueError(f"Factor {name!r}: availability_lag must be nonnegative")
        expressions.append(pl.col(name).shift(lag).over("stock").alias(name))
    return frame.with_columns(expressions)


def preprocess_factors(
    panel: pl.DataFrame,
    specs: Sequence[FactorSpec],
    cfg: Mapping[str, Any] | Any,
) -> pl.DataFrame:
    """Clean and standardize factors daily, preserving input rows and columns.

    Assume :func:`apply_availability_lags` has run on the complete daily table; do not
    shift again. Process Inf -> null, quantile winsorization, industry demeaning,
    cross-sectional regression residuals against ``log(market_cap)``, zscore/rank, then direction.
    """
    options = _section(cfg, "preprocess")
    custom_method = options.get("method")
    if custom_method:
        result = import_callable(str(custom_method))(panel, specs, cfg)
        if not isinstance(result, pl.DataFrame):
            raise TypeError("Custom preprocess methods must return a Polars DataFrame")
        expected = [
            f"factor__{_spec_value(spec, 'output_name', _spec_value(spec, 'name'))}"
            for spec in specs
        ]
        _require_columns(result, [*KEYS, *expected])
        return result
    _require_columns(panel, KEYS)
    lower = float(options.get("winsor_lower", 0.01))
    upper = float(options.get("winsor_upper", 0.99))
    if not 0 <= lower < upper <= 1:
        raise ValueError("winsor_lower / winsor_upper must satisfy 0 <= lower < upper <= 1")

    standardize = str(options.get("standardize", "zscore"))
    if standardize not in {"zscore", "rank", "none"}:
        raise ValueError("standardize must be zscore, rank, or none")

    industry_col = str(options.get("industry_column", "industry"))
    size_col = str(options.get("market_cap_column", "market_cap"))
    use_industry = bool(options.get("neutralize_industry", False))
    use_size = bool(options.get("neutralize_size", False))
    if use_industry:
        _require_columns(panel, [industry_col])
    if use_size:
        _require_columns(panel, [size_col])

    frame = panel
    aliases: set[str] = set()
    for spec in specs:
        raw = str(_spec_value(spec, "name"))
        alias = str(_spec_value(spec, "output_name", raw))
        direction = float(_spec_value(spec, "direction", 1.0))
        output = f"factor__{alias}"
        if output in aliases:
            raise ValueError(f"Duplicate factor output column: {output}")
        aliases.add(output)
        _require_columns(frame, [raw])

        # Carry each step through the output column and retain raw columns for debugging.
        value = pl.col(raw).cast(pl.Float64, strict=False)
        frame = frame.with_columns(
            pl.when(value.is_finite()).then(value).otherwise(None).alias(output)
        )
        frame = _winsorize(frame, output, lower, upper)
        if use_industry:
            frame = frame.with_columns(
                pl.when(pl.col(industry_col).is_not_null())
                .then(
                    pl.col(output)
                    - pl.col(output).mean().over(["date", industry_col])
                )
                .otherwise(None)
                .alias(output)
            )
        if use_size:
            frame = _neutralize_size(
                frame,
                output,
                size_col,
                group_col=industry_col if use_industry else None,
            )
        frame = _standardize(frame, output, standardize)
        frame = frame.with_columns((pl.col(output) * direction).alias(output))
    return frame


def combine_scores(
    panel: pl.DataFrame,
    specs: Sequence[FactorSpec],
    cfg: Mapping[str, Any] | Any,
) -> pl.DataFrame:
    """Combine factors into ``score`` without dropping stocks because one factor is missing.

    Built-ins: ``weighted``, ``equal``, ``rank_average``. Configure custom methods as
    ``module:function``; the function receives ``(panel, specs, cfg)`` and returns a
    DataFrame with ``score``, a Series named score, or a Polars Expr.
    """
    combine = _section(cfg, "combine")
    preprocess = _section(cfg, "preprocess")
    method = str(combine.get("method", "weighted"))
    if ":" in method:
        return _custom_score(import_callable(method)(panel, specs, cfg), panel)
    if method not in {"weighted", "equal", "rank_average"}:
        raise ValueError(f"Unknown factor combination method: {method!r}")

    columns = [
        f"factor__{_spec_value(spec, 'output_name', _spec_value(spec, 'name'))}"
        for spec in specs
    ]
    _require_columns(panel, ["date", *columns])
    weights = [float(_spec_value(spec, "weight", 1.0)) for spec in specs]
    if method == "weighted" and not any(weights):
        raise ValueError("weighted combination requires at least one nonzero weight")

    missing = str(combine.get("missing", preprocess.get("missing", "renormalize")))
    min_count = int(
        combine.get("min_factor_count", preprocess.get("min_factor_count", 1))
    )
    if missing not in {"renormalize", "require_all", "min_count"}:
        raise ValueError("missing must be renormalize, require_all, or min_count")
    if min_count < 1:
        raise ValueError("min_factor_count must be positive")

    frame = panel
    temporary_columns: list[str] = []
    if method == "rank_average":
        ranked = _temporary_names(frame, len(columns))
        for column, temp in zip(columns, ranked):
            frame = frame.with_columns(_rank_score(column).alias(temp))
        columns = ranked
        temporary_columns = ranked
        weights = [1.0] * len(columns)
        method = "equal"

    valid_count = pl.sum_horizontal(
        [pl.col(column).is_not_null().cast(pl.Int16) for column in columns]
    )
    if method == "weighted":
        numerator = pl.sum_horizontal(
            [pl.col(column) * weight for column, weight in zip(columns, weights)]
        )
        denominator = pl.sum_horizontal(
            [
                pl.when(pl.col(column).is_not_null())
                .then(abs(weight))
                .otherwise(0.0)
                for column, weight in zip(columns, weights)
            ]
        )
    else:
        numerator = pl.sum_horizontal([pl.col(column) for column in columns])
        denominator = valid_count.cast(pl.Float64)

    required = len(columns) if missing == "require_all" else min_count
    score = (
        pl.when((valid_count >= required) & (denominator > 0))
        .then(numerator / denominator)
        .otherwise(None)
        .alias("score")
    )
    result = frame.with_columns(score)
    if temporary_columns:
        result = result.drop(temporary_columns)
    return result


def _winsorize(
    frame: pl.DataFrame, column: str, lower: float, upper: float
) -> pl.DataFrame:
    low = pl.col(column).quantile(lower, interpolation="linear").over("date")
    high = pl.col(column).quantile(upper, interpolation="linear").over("date")
    return frame.with_columns(
        pl.when(pl.col(column) < low)
        .then(low)
        .when(pl.col(column) > high)
        .then(high)
        .otherwise(pl.col(column))
        .alias(column)
    )


def _neutralize_size(
    frame: pl.DataFrame,
    column: str,
    size_col: str,
    group_col: str | None = None,
) -> pl.DataFrame:
    """Compute daily size residuals; use joint FWL neutralization when industry data is available."""
    names = _temporary_names(frame, 7)
    size, paired, paired_size, x_center, s_center, numerator, denominator = names
    raw_size = pl.col(size_col).cast(pl.Float64, strict=False)
    frame = frame.with_columns(
        pl.when((raw_size > 0) & raw_size.is_finite())
        .then(raw_size.log())
        .otherwise(None)
        .alias(size)
    )
    frame = frame.with_columns(
        pl.when(pl.col(size).is_not_null() & pl.col(column).is_not_null())
        .then(pl.col(column))
        .otherwise(None)
        .alias(paired),
        pl.when(pl.col(size).is_not_null() & pl.col(column).is_not_null())
        .then(pl.col(size))
        .otherwise(None)
        .alias(paired_size),
    )
    groups = ["date", group_col] if group_col else ["date"]
    frame = frame.with_columns(
        (pl.col(paired) - pl.col(paired).mean().over(groups)).alias(x_center),
        (pl.col(paired_size) - pl.col(paired_size).mean().over(groups)).alias(
            s_center
        ),
    )
    frame = frame.with_columns(
        (pl.col(x_center) * pl.col(s_center)).sum().over("date").alias(numerator),
        (pl.col(s_center) ** 2).sum().over("date").alias(denominator),
    )
    beta = pl.when(pl.col(denominator).abs() > 1e-14).then(
        pl.col(numerator) / pl.col(denominator)
    ).otherwise(0.0)
    return frame.with_columns(
        pl.when(pl.col(paired).is_not_null())
        .then(pl.col(x_center) - beta * pl.col(s_center))
        .otherwise(None)
        .alias(column)
    ).drop(names)


def _standardize(frame: pl.DataFrame, column: str, method: str) -> pl.DataFrame:
    if method == "none":
        return frame
    if method == "rank":
        return frame.with_columns(_rank_score(column).alias(column))
    mean = pl.col(column).mean().over("date")
    std = pl.col(column).std().over("date")
    return frame.with_columns(
        pl.when(std.abs() > 1e-14)
        .then((pl.col(column) - mean) / std)
        .otherwise(None)
        .alias(column)
    )


def _rank_score(column: str) -> pl.Expr:
    rank = pl.col(column).rank(method="average").over("date").cast(pl.Float64)
    count = pl.col(column).count().over("date")
    return pl.when(count > 1).then(2.0 * (rank - 1.0) / (count - 1.0) - 1.0).when(
        count == 1
    ).then(0.0).otherwise(None)


def _custom_score(result: Any, panel: pl.DataFrame) -> pl.DataFrame:
    if isinstance(result, pl.Expr):
        return panel.with_columns(result.alias("score"))
    if isinstance(result, pl.Series):
        if len(result) != panel.height:
            raise ValueError("Custom score Series length does not match panel")
        return panel.with_columns(result.alias("score"))
    if isinstance(result, pl.DataFrame):
        if "score" not in result.columns:
            raise ValueError("Custom combination DataFrame must contain score")
        if result.height == panel.height and not all(key in result.columns for key in KEYS):
            return panel.with_columns(result["score"])
        _require_columns(result, [*KEYS, "score"])
        return panel.join(result.select(*KEYS, "score"), on=KEYS, how="left")
    raise TypeError("Custom combination methods must return DataFrame, Series, or Expr")


def _section(cfg: Mapping[str, Any] | Any, name: str) -> Mapping[str, Any]:
    if hasattr(cfg, name):
        value = getattr(cfg, name)
        if isinstance(value, Mapping):
            return value
    if isinstance(cfg, Mapping):
        value = cfg.get(name)
        return value if isinstance(value, Mapping) else cfg
    raise TypeError("cfg must be Config or dict")


def _spec_value(spec: FactorSpec | Mapping[str, Any], name: str, default: Any = None) -> Any:
    if name == "output_name" and hasattr(spec, "output_name"):
        return spec.output_name
    if isinstance(spec, Mapping):
        if name == "output_name":
            return spec.get("alias") or spec.get("name", default)
        return spec.get(name, default)
    return getattr(spec, name, default)


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def _temporary_names(frame: pl.DataFrame, count: int) -> list[str]:
    names = []
    for index in range(count):
        name = f"__ashare_tmp_{index}"
        while name in frame.columns or name in names:
            name += "_"
        names.append(name)
    return names
