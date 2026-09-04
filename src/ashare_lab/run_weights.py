from pathlib import Path

import polars as pl

from ashare_lab.charts import write_backtest_charts
from ashare_lab.execution import run_execution
from ashare_lab.pipeline import make_catalogs
from ashare_lab.plugins import import_callable
from ashare_lab.settings import load_config


EXCEL_FILE = "中证1000成分股.xlsx"
CONFIG_FILE = "configs/csi1000.toml"
OUTPUT_DIR = Path("outputs/csi1000_excel_weights")


# 1. 读取 TOML：回测区间、成交方式、规则、费用都由它控制
config = load_config(CONFIG_FILE)


# 2. 读取 Excel
weights = pl.read_excel(
    EXCEL_FILE,
    sheet_name="Sheet1",
    engine="calamine",
)

# 正常情况下 Excel 日期会被识别为 Date。
# 如果读取成 Excel 序号，则转换为真实日期。
if weights.schema["日期"].is_numeric():
    date_expression = (
        pl.date(1899, 12, 30)
        + pl.duration(days=pl.col("日期").cast(pl.Int64))
    )
else:
    date_expression = pl.col("日期").cast(pl.Date)

weights = weights.select(
    date_expression.alias("date"),
    pl.col("股票代码").cast(pl.String).alias("stock"),
    pl.col("权重").cast(pl.Float64).alias("target_weight"),
)

# 只使用 TOML 规定的研究区间
weights = weights.filter(
    pl.col("date").is_between(config.start, config.end)
)

if weights.is_empty():
    raise ValueError("TOML 的研究区间内没有 Excel 权重")

# 保留 Excel 中的原始权重，不做归一化。


# 3. 读取这些股票的行情和涨跌停状态
catalog, _ = make_catalogs(config)
stocks = weights["stock"].unique().to_list()

market = catalog.market(
    columns=["open", "close", "adj_open", "adj_close", "trade_status"],
    start=config.start,
    end=config.end,
    stocks=stocks,
)

limits = catalog.valuation(
    columns=["limit_status"],
    start=config.start,
    end=config.end,
    stocks=stocks,
)

market = (
    market.join(limits, on=["date", "stock"], how="left")
    .rename({
        "open": "raw_open",
        "close": "raw_close",
    })
)


# 4. 删除区间内无法成交的调仓日
# next_open 要求信号日后还有一个交易日
calendar = market["date"].unique().sort().to_list()

if config.execution["price"] == "next_open":
    executable_signal_dates = calendar[:-1]
else:
    executable_signal_dates = calendar

weights = weights.filter(
    pl.col("date").is_in(executable_signal_dates)
)

if weights.is_empty():
    raise ValueError("区间内没有可以完成成交的调仓日")


# 5. 加载可选的整手插件
lot_method = config.execution.get("lot_rounder")
lot_rounder = (
    import_callable(str(lot_method))
    if lot_method
    else None
)


# 6. 运行撮合
result = run_execution(
    market,
    weights,
    config=config,
    tradability_rule=config.rules.get("method"),
    lot_rounder=lot_rounder,
)


# 7. 生成净值并保存
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 旧版本保存的是归一化权重，删除它以免和当前原始权重文件混淆。
legacy_targets = OUTPUT_DIR / "normalized_targets.parquet"
if legacy_targets.exists():
    legacy_targets.unlink()

nav = result.daily_nav.with_columns(
    (pl.col("nav") / pl.col("initial_nav")).alias("net_value")
)

nav.write_csv(OUTPUT_DIR / "nav.csv")
write_backtest_charts(nav, OUTPUT_DIR, config)
result.trades.write_parquet(OUTPUT_DIR / "trades.parquet")
result.positions.write_parquet(OUTPUT_DIR / "positions.parquet")
result.rejections.write_parquet(OUTPUT_DIR / "rejections.parquet")
weights.write_parquet(OUTPUT_DIR / "targets.parquet")

print(
    nav.select(
        "date",
        "net_value",
        "daily_return",
        "turnover",
        "fees",
    ).tail(10)
)

print(f"\n调仓日数量：{weights['date'].n_unique()}")
weight_sums = weights.group_by("date").agg(pl.col("target_weight").sum())
print(
    "原始权重合计范围："
    f"{weight_sums['target_weight'].min():.6f} ~ "
    f"{weight_sums['target_weight'].max():.6f}"
)
print(f"成交笔数：{result.trades.height}")
print(f"拒单笔数：{result.rejections.height}")
print(f"最终净值：{nav['net_value'][-1]:.6f}")
print(f"结果目录：{OUTPUT_DIR}")
