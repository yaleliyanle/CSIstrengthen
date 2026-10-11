from pathlib import Path

import polars as pl

from ashare_lab.charts import write_backtest_charts
from ashare_lab.execution import run_execution
from ashare_lab.pipeline import make_catalogs
from ashare_lab.plugins import import_callable
from ashare_lab.settings import load_config


# Keep the local workbook name and its original Date/Stock Code/Weight headers.
EXCEL_FILE = "中证1000成分股.xlsx"
CONFIG_FILE = "configs/csi1000.toml"
OUTPUT_DIR = Path("outputs/csi1000_excel_weights")


# 1. Load TOML: it controls the backtest interval, execution, rules, and costs
config = load_config(CONFIG_FILE)


# 2. Read Excel
weights = pl.read_excel(
    EXCEL_FILE,
    sheet_name="Sheet1",
    engine="calamine",
)

# Excel dates are normally recognized as Date.
# Convert Excel serial numbers to actual dates if needed.
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

# Use only the research interval specified in TOML
weights = weights.filter(
    pl.col("date").is_between(config.start, config.end)
)

if weights.is_empty():
    raise ValueError("No Excel weights within the TOML research interval")

# Preserve the original Excel weights without normalization.


# 3. Read market data and price-limit status for these stocks
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


# 4. Drop rebalance dates that cannot execute within the interval
# next_open requires a trading day after the signal date
calendar = market["date"].unique().sort().to_list()

if config.execution["price"] == "next_open":
    executable_signal_dates = calendar[:-1]
else:
    executable_signal_dates = calendar

weights = weights.filter(
    pl.col("date").is_in(executable_signal_dates)
)

if weights.is_empty():
    raise ValueError("No executable rebalance dates within the interval")


# 5. Load the optional board-lot plugin
lot_method = config.execution.get("lot_rounder")
lot_rounder = (
    import_callable(str(lot_method))
    if lot_method
    else None
)


# 6. Run execution
result = run_execution(
    market,
    weights,
    config=config,
    tradability_rule=config.rules.get("method"),
    lot_rounder=lot_rounder,
)


# 7. Generate and save NAV
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove legacy normalized weights to avoid confusion with the current raw-weight output.
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

print(f"\nNumber of rebalance dates: {weights['date'].n_unique()}")
weight_sums = weights.group_by("date").agg(pl.col("target_weight").sum())
print(
    "Range of total raw weights: "
    f"{weight_sums['target_weight'].min():.6f} ~ "
    f"{weight_sums['target_weight'].max():.6f}"
)
print(f"Number of trades: {result.trades.height}")
print(f"Number of rejected orders: {result.rejections.height}")
print(f"Final NAV: {nav['net_value'][-1]:.6f}")
print(f"Output directory: {OUTPUT_DIR}")
