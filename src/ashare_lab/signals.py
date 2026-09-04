"""因子滞后、截面预处理与合成。

约定很少：键名固定为 ``date`` / ``stock``，处理后的因子名为
``factor__<alias>``。新增合成方法时也可以直接在配置中写
``your_module:your_function``。
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
    """按股票把原始因子向后移动其可用滞后日数。

    这里的“一日”是一只股票在输入中的一行。因此必须先对**完整连续的
    日频因子表**调用本函数，再筛选周度/月度调仓日；若先筛调仓日，lag=1
    就会错误地变成滞后一整个调仓周期。

    原始因子列会被覆盖，行数不变。
    """
    _require_columns(daily_factors, KEYS)
    names = [_spec_value(spec, "name") for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("apply_availability_lags 不支持重复的因子 name")
    _require_columns(daily_factors, names)

    frame = daily_factors.sort(KEYS[::-1])  # stock，再 date
    expressions = []
    for spec in specs:
        name = str(_spec_value(spec, "name"))
        lag = int(_spec_value(spec, "availability_lag", 0))
        if lag < 0:
            raise ValueError(f"因子 {name!r} 的 availability_lag 不能为负数")
        expressions.append(pl.col(name).shift(lag).over("stock").alias(name))
    return frame.with_columns(expressions)


def preprocess_factors(
    panel: pl.DataFrame,
    specs: Sequence[FactorSpec],
    cfg: Mapping[str, Any] | Any,
) -> pl.DataFrame:
    """逐日清洗并标准化因子，保留输入行与输入列。

    本函数假定 :func:`apply_availability_lags` 已在完整日频表上执行，不会
    再次 shift。处理顺序是：Inf→null、分位数缩尾、行业去均值、对
    ``log(market_cap)`` 做截面线性回归取残差、zscore/rank、应用方向。
    """
    options = _section(cfg, "preprocess")
    custom_method = options.get("method")
    if custom_method:
        result = import_callable(str(custom_method))(panel, specs, cfg)
        if not isinstance(result, pl.DataFrame):
            raise TypeError("自定义 preprocess 方法必须返回 Polars DataFrame")
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
        raise ValueError("winsor_lower / winsor_upper 应满足 0 <= lower < upper <= 1")

    standardize = str(options.get("standardize", "zscore"))
    if standardize not in {"zscore", "rank", "none"}:
        raise ValueError("standardize 只能是 zscore、rank 或 none")

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
            raise ValueError(f"重复的因子输出列: {output}")
        aliases.add(output)
        _require_columns(frame, [raw])

        # 用输出列承接每一步，原始列始终保留，便于排查数据。
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
    """合成为 ``score``，不因某个因子缺失而删除股票。

    内置 ``weighted``、``equal``、``rank_average``。自定义方法写成
    ``module:function``，函数接收 ``(panel, specs, cfg)``，并返回带
    ``score`` 的 DataFrame、名为 score 的 Series，或一个 Polars Expr。
    """
    combine = _section(cfg, "combine")
    preprocess = _section(cfg, "preprocess")
    method = str(combine.get("method", "weighted"))
    if ":" in method:
        return _custom_score(import_callable(method)(panel, specs, cfg), panel)
    if method not in {"weighted", "equal", "rank_average"}:
        raise ValueError(f"未知因子合成方法: {method!r}")

    columns = [
        f"factor__{_spec_value(spec, 'output_name', _spec_value(spec, 'name'))}"
        for spec in specs
    ]
    _require_columns(panel, ["date", *columns])
    weights = [float(_spec_value(spec, "weight", 1.0)) for spec in specs]
    if method == "weighted" and not any(weights):
        raise ValueError("weighted 合成至少需要一个非零权重")

    missing = str(combine.get("missing", preprocess.get("missing", "renormalize")))
    min_count = int(
        combine.get("min_factor_count", preprocess.get("min_factor_count", 1))
    )
    if missing not in {"renormalize", "require_all", "min_count"}:
        raise ValueError("missing 只能是 renormalize、require_all 或 min_count")
    if min_count < 1:
        raise ValueError("min_factor_count 必须大于 0")

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
    """逐日取市值残差；有行业时使用 FWL 联合中性化。"""
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
            raise ValueError("自定义 score Series 长度与 panel 不一致")
        return panel.with_columns(result.alias("score"))
    if isinstance(result, pl.DataFrame):
        if "score" not in result.columns:
            raise ValueError("自定义合成 DataFrame 必须包含 score")
        if result.height == panel.height and not all(key in result.columns for key in KEYS):
            return panel.with_columns(result["score"])
        _require_columns(result, [*KEYS, "score"])
        return panel.join(result.select(*KEYS, "score"), on=KEYS, how="left")
    raise TypeError("自定义合成方法应返回 DataFrame、Series 或 Expr")


def _section(cfg: Mapping[str, Any] | Any, name: str) -> Mapping[str, Any]:
    if hasattr(cfg, name):
        value = getattr(cfg, name)
        if isinstance(value, Mapping):
            return value
    if isinstance(cfg, Mapping):
        value = cfg.get(name)
        return value if isinstance(value, Mapping) else cfg
    raise TypeError("cfg 应为 Config 或 dict")


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
        raise ValueError(f"缺少列: {', '.join(missing)}")


def _temporary_names(frame: pl.DataFrame, count: int) -> list[str]:
    names = []
    for index in range(count):
        name = f"__ashare_tmp_{index}"
        while name in frame.columns or name in names:
            name += "_"
        names.append(name)
    return names
