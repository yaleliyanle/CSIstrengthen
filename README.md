# AShare Lab

An A-share multifactor framework for CSI 1000 index-enhancement research. It focuses on three tasks:

1. Read only the years, stocks, and columns needed for a run from the 51 GB dataset.
2. Separate factor cleaning, combination, stock selection, rebalancing, execution, trading rules, and costs.
3. Select common variations from the command line and implement specialized methods as small function plugins.

Polars is the default backend. DuckDB is also available for Parquet scanning, date/stock filtering, joins, and batched factor loading. Both backends share the same research and backtesting logic.

Research data, generated outputs, and local document exports are excluded from Git. Supply the datasets locally before running data-dependent commands. Original Chinese dataset filenames, factor-source directory names, Excel column names, and trading-status values are retained as data-interface identifiers. Documentation, comments, CLI help, and diagnostic messages are in English.

| Original local identifier | Meaning |
|---|---|
| `a share/A股行情数据.parquet` | A-share market data |
| `a share/A股估值数据.parquet` | A-share valuation data |
| `a share/A股行业分类.parquet` | A-share industry classifications |
| `a share/指数成分股.parquet` | Historical index constituents |
| `基本面因子` | Fundamental factors |
| `低频量价` | Low-frequency price/volume factors |
| `高频量价` | High-frequency price/volume factors |
| `一致预期因子` | Consensus-estimate factors |
| `深度学习因子` | Deep-learning factors |
| `中证1000成分股.xlsx` | Local CSI 1000 constituent-weight workbook |
| `日期`, `股票代码`, `权重` | Date, stock code, and weight columns in the local workbook |
| `交易`, `停牌`, `停市` | Trading, suspended, and halted status values |

Change data paths in TOML as needed. Factor sources must match directory names under `factor_root`.

## 1. Installation and initial checks

Use `uv` from the project root:

```bash
uv sync --group dev
uv run ashare-lab --help
uv run ashare-lab data-info --config configs/csi1000.toml
```

Keep using `uv run ashare-lab ...`; no manual virtual-environment activation is needed.

### Switching between Polars and DuckDB

Defaults are in `configs/csi1000.toml`:

```toml
[data]
engine = "polars"       # Change to duckdb to switch backends
batch_size = 16         # Factors per DuckDB batch
duckdb_threads = 4
```

Override the backend in the VS Code terminal without editing TOML:

```bash
# Use the original Polars backend
uv run ashare-lab validate-factors --config configs/csi1000.toml \
  --data-engine polars

# Use DuckDB with the same other settings and outputs
uv run ashare-lab validate-factors --config configs/csi1000.toml \
  --data-engine duckdb
```

`--data-engine` works with `single-factor`, `select-factors`, `validate-factors`, and `backtest`. DuckDB replaces data access while preserving neutralization, IC, Lasso, trading rules, and NAV calculations.

Inspect available factors:

```bash
# List all 1,152 factors
uv run ashare-lab list-factors --config configs/csi1000.toml

# List fundamental factors using the original source-directory identifier
uv run ashare-lab list-factors --config configs/csi1000.toml --source 基本面因子

# Search factor names containing hist_vol
uv run ashare-lab list-factors --config configs/csi1000.toml --contains hist_vol
```

## 2. Common backtest commands

### Weekly rebalancing, next-day open, standard A-share rules

```bash
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rebalance weekly \
  --execution next_open \
  --rules standard
```

Generate target weights on each week's last actual trading day and execute at the next trading day's open. Apply suspension restrictions, T+1, and configured costs. Daily data cannot establish whether a stock was locked at its price limit at the opening instant, so end-of-day price-limit fields do not block opening orders by default.

### Monthly rebalancing, same-day close, minimal rules

```bash
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rebalance monthly \
  --execution same_close \
  --rules minimal \
  --set execution.allow_same_close=true
```

Execute at the close of each month's last actual trading day. Require an execution price, disable suspension/price-limit restrictions and T+1, and set costs to zero.

`same_close` assumes signals are available before closing execution. Complete same-day closing data may introduce look-ahead bias. The framework warns by default; `allow_same_close=true` acknowledges this assumption and suppresses the warning.

### Monthly rebalancing, next-day open, one factor

```bash
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --start 2022-01-01 \
  --end 2025-12-31 \
  --rebalance monthly \
  --execution next_open \
  --rules minimal \
  --factor '基本面因子:bp_lf:1:1' \
  --set preprocess.min_factor_count=1 \
  --set portfolio.method=top_n_equal \
  --set portfolio.top_n=100
```

`--factor` uses this format:

```text
SOURCE_DIRECTORY:FACTOR_NAME:DIRECTION:COMBINATION_WEIGHT
```

- Direction `1`: higher values are better.
- Direction `-1`: lower values are better.
- Direction and weight default to `1` when omitted.
- Quote the whole argument when a factor name contains `+`.

### Combining factors from the command line

```bash
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --factor '基本面因子:bp_lf:1:1' \
  --factor '基本面因子:roe_ttm:1:1' \
  --factor '低频量价:hist_vol_20d:-1:1' \
  --rebalance weekly \
  --execution next_open \
  --rules standard
```

Clean and standardize each factor independently within each cross-section, then combine by weight to avoid adding incompatible scales.

## 3. Single-factor diagnostics

```bash
uv run ashare-lab single-factor \
  --config configs/csi1000.toml \
  --factor '基本面因子:bp_lf:1:1' \
  --rebalance monthly \
  --execution next_open \
  --set analytics.horizons='[1,5,20]' \
  --output outputs/bp_test
```

Outputs include:

- Factor coverage.
- Pearson IC and RankIC for each period.
- Mean IC, ICIR, and t-statistics.
- Five quantile returns and top-minus-bottom returns.
- Quantile-portfolio turnover.
- Daily details and summary CSVs for each holding horizon.

For `next_open`, `forward_5d` enters at the next trading day's open and exits at the open five market trading days later. Missing stock rows do not redefine five days as that stock's next five available records.

When holding periods overlap adjacent signals, such as daily signals with 20-day returns, ICIR and t-statistics use Newey-West/HAC adjustment with automatic lag selection. Summaries retain independent-sample measures with the `_naive` suffix. Use `--set analytics.hac_lags=0` to disable adjustment or a nonnegative integer to set the lag order.

### Batch factor selection

Use a historical interval for training. This command produces a selection table without automatically backtesting:

```bash
uv run ashare-lab select-factors \
  --config configs/csi1000.toml \
  --source 基本面因子 \
  --start 2014-10-31 \
  --end 2020-12-31 \
  --rebalance monthly \
  --method ic \
  --horizon 20 \
  --output outputs/factor_ic
```

Switch to Lasso by changing the method and penalty:

```bash
uv run ashare-lab select-factors \
  --config configs/csi1000.toml \
  --source 基本面因子 \
  --start 2014-10-31 \
  --end 2020-12-31 \
  --rebalance monthly \
  --method lasso \
  --set selection.alpha=0.02 \
  --output outputs/factor_lasso
```

Results are in `<output>/factor_selection/selection.csv`. Separate selection and subsequent backtest intervals to avoid selecting factors using future returns.

### Two-stage selection: fixed splits and rolling validation

Fixed splits follow `[split]`: single-factor IC -> Lasso -> parameter selection on validation data. The process does not read `test/live` intervals:

```bash
uv run ashare-lab validate-factors \
  --config configs/csi1000.toml \
  --source 基本面因子 \
  --rebalance monthly \
  --split fixed \
  --output outputs/fixed_validation
```

Rolling validation uses the same out-of-sample boundaries and creates folds only inside the train + validation development interval:

```bash
uv run ashare-lab validate-factors \
  --config configs/csi1000.toml \
  --source 基本面因子 \
  --rebalance monthly \
  --split rolling \
  --rolling-window expanding \
  --rolling-train-months 24 \
  --rolling-validation-months 6 \
  --rolling-step-months 6 \
  --output outputs/rolling_validation
```

Set dates in TOML or override directly:

```bash
uv run ashare-lab validate-factors \
  --config configs/csi1000.toml \
  --source 基本面因子 \
  --split fixed \
  --train-start 2014-10-31 --train-end 2019-04-30 \
  --validation-start 2019-05-01 --validation-end 2020-10-31 \
  --test-start 2020-11-01 --test-end 2023-10-31 \
  --live-start 2023-11-01 --live-end 2026-06-30 \
  --output outputs/custom_split
```

The program purges trailing signals whose return labels cross interval boundaries, using `selection.horizon` and execution timing. Candidate parameters come from `selection.ic_thresholds` and `selection.alpha_grid`.

Results write to `<output>/factor_validation/`: `split.csv` and `folds.csv` audit time splits; `candidates.csv` compares parameters; `factor_stability.csv` checks direction stability; `selected_factors.csv` contains the final selection.

## 4. Three trading-rule profiles

| Profile | Suspensions / price limits | T+1 | Costs | Use case |
|---|---|---|---|---|
| `minimal` | No blocking; missing prices always prevent execution | Disabled | Forced to zero | Inspect signal potential and debug research logic |
| `standard` | Suspensions block both sides; upper limits block buys and lower limits block sells by default | Enabled by default | Historical costs by default | Regular A-share research with `--set` overrides |
| `custom` | Configure individually; plugins supported | Configure individually | Configure individually / plugin | Specialized accounts or execution assumptions |

`minimal` forces suspension/price-limit blocking, T+1, and all costs off even with conflicting overrides. `standard` supplies default market rules with individual overrides. `custom` shares those defaults and makes per-rule configuration explicit:

```bash
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rules custom \
  --set rules.block_suspended=true \
  --set rules.block_limit=false \
  --set rules.t_plus_one=false \
  --set costs.commission_rate=0.00015 \
  --set costs.minimum_commission=0 \
  --set costs.stamp_tax=0.0005 \
  --set costs.transfer_fee=none
```

Cost options support:

- `historical`: built-in historical rates.
- `none`: disable the component.
- A number: fixed rate, such as `0.0005`.
- `module:function`: custom cost logic.

Inspect the resolved configuration:

```bash
uv run ashare-lab show-config \
  --config configs/csi1000.toml \
  --rebalance monthly \
  --execution same_close \
  --rules minimal
```

## 5. Other command-line settings

Override any TOML field with `--set path=value`; repeat as needed.

| Desired behavior | Option |
|---|---|
| Weekly / monthly | `--rebalance weekly` / `--rebalance monthly` |
| Every 10 trading days | `--rebalance every_n_days --set schedule.every_n_days=10` |
| Same-day close / next-day open | `--execution same_close` / `--execution next_open` |
| Minimal / standard / custom rules | `--rules minimal` / `standard` / `custom` |
| Backtest interval | `--start 2020-01-01 --end 2025-12-31` |
| Index code | `--set research.benchmark=000852.SH` |
| Lag constituent snapshots by one trading day | `--set research.membership_lag=1` |
| Equal-weight top N | `--set portfolio.method=top_n_equal --set portfolio.top_n=100` |
| Index-weight top N | `--set portfolio.method=top_n_benchmark` |
| Tilt index weights | `--set portfolio.method=benchmark_tilt --set portfolio.tilt=0.5` |
| Disable industry neutralization | `--set preprocess.neutralize_industry=false` |
| Disable size neutralization | `--set preprocess.neutralize_size=false` |
| Require two available factors | `--set preprocess.min_factor_count=2` |
| Stop / hold / move to cash when scores are invalid | `--set portfolio.empty_signal=error` / `hold` / `cash` |
| Quantile count | `--set analytics.quantiles=10` |
| Output directory | `--output outputs/my_run` |
| Polars / DuckDB | `--data-engine polars` / `--data-engine duckdb` |
| Factors per DuckDB batch | `--set data.batch_size=16` |
| DuckDB query threads | `--set data.duckdb_threads=4` |
| Board-lot example | `--set execution.lot_rounder=ashare_lab.execution:round_to_board_lot` |

See [`configs/csi1000.toml`](configs/csi1000.toml) for the complete configuration. Copy a TOML file for recurring strategies and use overrides for temporary experiments.

`portfolio.empty_signal` defaults to `error` so missing data is not silently interpreted as holding positions. Choose `hold` or `cash` when that is the intended research assumption.

`round_to_board_lot` rounds buys and sells down to multiples of 100 shares. Implement details such as selling odd lots from corporate actions in an `execution.lot_rounder` plugin. By default, shares are not rounded, supporting initial weight-only research.

The engine accepts one complete target portfolio per day and trades each stock's net difference, selling before buying. This does not create same-day buy-then-sell trades in the same stock. The explicit `T+1` switch supports future intraday or multiple-batch extensions; it usually does not affect single-batch weekly/monthly results.

## 6. Output files

Backtests write to `<output>/backtest/`:

| File | Contents |
|---|---|
| `summary.json` | Strategy, benchmark, active returns, and run summary |
| `nav.csv` | Aligned strategy, benchmark, and excess NAV |
| `nav.svg` | Strategy, benchmark, and excess NAV curves |
| `drawdown.svg` | Strategy, benchmark, and excess NAV drawdowns |
| `strategy_daily.csv` | Daily NAV, cash, position value, one-way turnover, gross traded-value turnover, and costs |
| `benchmark_daily.csv` | Benchmark daily NAV, cash, position value, turnover, and costs |
| `trades.parquet` | Individual trades and costs |
| `rejections.parquet` | Orders rejected for suspensions, price limits, missing prices, cash, etc. |
| `positions.parquet` | Actual positions, recorded on rebalance dates by default |
| `targets.parquet` | Target weights |
| `signal_panel.parquet` | Rebalance-date factors, standardized values, and combined scores |
| `resolved_config.json` | Complete configuration for reproducibility |

Single-factor results write to `<output>/single_factor/`.

Factor selection writes to `<output>/factor_selection/`:

| File | Contents |
|---|---|
| `selection.csv` | Coverage, IC or Lasso coefficients, and selection status for each factor |
| `selected_factors.txt` | Selected aliases, one per line |
| `resolved_config.json` | Complete configuration used by the run |

## 7. Code structure: where to add a method

| Component | File | Entry point |
|---|---|---|
| Raw fields, loading, as-of constituents | `src/ashare_lab/data.py` | `DataCatalog` / `FactorCatalog` |
| DuckDB scanning, filtering, joins, batching | `src/ashare_lab/duckdb_data.py` | `DuckDBDataCatalog` / `DuckDBFactorCatalog` |
| Lags, winsorization, neutralization, standardization | `src/ashare_lab/signals.py` | `preprocess_factors` |
| IC, Lasso, custom selection | `src/ashare_lab/selection.py` | `select_factors` |
| NAV/drawdown charts, custom plotting | `src/ashare_lab/charts.py` | `write_backtest_charts` |
| Factor combination | `src/ashare_lab/signals.py` | `combine_scores` |
| Weekly/monthly/custom dates | `src/ashare_lab/schedule.py` | `make_schedule` |
| Top N, index tilts | `src/ashare_lab/portfolio.py` | `build_targets` |
| Public execution interface | `src/ashare_lab/execution.py` | `run_execution` |
| Commission, stamp duty, transfer fees, custom costs | `src/ashare_lab/costs.py` | `HistoricalAshareCosts` / `CostContext` |
| Execution settings, validation, date mapping | `src/ashare_lab/execution_inputs.py` | `ExecutionConfig` / `prepare_market` |
| Sell-before-buy logic, restrictions, cash, accounting | `src/ashare_lab/execution_engine.py` | `run_engine` |
| IC, quantile returns | `src/ashare_lab/analytics.py` | `analyze_factor` |
| NAV metrics | `src/ashare_lab/metrics.py` | `performance_metrics` |
| Pipeline orchestration | `src/ashare_lab/pipeline.py` | `run_factor_research` / `run_backtest` |
| CLI arguments | `src/ashare_lab/cli.py` | `build_parser` |

The main flow is:

```text
Read selected data columns
  -> point-in-time CSI 1000 constituents
  -> information-availability lags
  -> independent factor preprocessing
  -> factor combination
  -> target weights
  -> open/close execution and rules
  -> NAV, trades, rejections, reports
```

## 8. Plugins: add methods through configuration

Configure `module:function` to load a plugin. [`src/ashare_lab/example_plugins.py`](src/ashare_lab/example_plugins.py), included in the package, contains minimal working examples to copy and adapt:

```bash
# Portfolio: equally weight the highest-scoring 20%
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --set portfolio.method=ashare_lab.example_plugins:top_twenty_percent

# Combination: equally average available standardized factors
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --set combine.method=ashare_lab.example_plugins:average_available_factors

# Schedule: first trading day of each month
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --set schedule.method=ashare_lab.example_plugins:first_trading_day_each_month

# Tradability: reject new-stock orders with trade_status=N
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rules custom \
  --set rules.method=ashare_lab.example_plugins:block_new_stocks

# Full cost model: zero costs
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rules custom \
  --set costs.method=ashare_lab.example_plugins:zero_cost
```

Each plugin follows an input/output contract:

| Component | Input | Output |
|---|---|---|
| Schedule | `calendar, config` | `list[date]` drawn from the supplied calendar |
| Full preprocessing | `panel, factor_specs, config` | Polars `DataFrame` with `date, stock, factor__<alias>` |
| Combination | `panel, factor_specs, config` | `DataFrame` with `score`, or `Series` / `Expr` |
| Portfolio | `panel, config` | `DataFrame` with `date, stock, target_weight`; nonnegative weights summing to 1 each period |
| Tradability | One `OrderContext` | `None/True` allows execution; `False` or a string rejects |
| Full cost model | `date, side, value`, or one `CostContext` | `TradeCost`; an object with `calculate(...)` also works |
| Board lots | `requested_shares, side, stock, date` | Actual share quantity |

Keys are `schedule.method`, `preprocess.method`, `combine.method`, `portfolio.method`, `rules.method`, `costs.method`, and `execution.lot_rounder`.

`costs.method` replaces the whole cost model and returns `TradeCost`. `costs.stamp_tax` and `costs.transfer_fee` replace individual components and return fee amounts as `float`. Keep these signatures distinct.

For exchange differences, early transfer fees, or market impact, define the full cost function with one `CostContext` to access `context.stock/date/side/value`. Built-in transfer fees cover the unified convention and 2022-04-29 boundary; add exchange differences through this interface for earlier history.

## 9. Dataset adaptation and look-ahead controls

The framework follows the actual Parquet structure rather than assumptions hard-coded from two text files:

- Market data is a long stock panel from 2004-12-31 to 2026-05-27. Derive trading days exclusively from it.
- Valuation and industry data use calendar days and cannot define the trading calendar.
- CSI 1000 `000852.SH` has month-end snapshots starting on 2014-10-31, not daily constituents. Use the latest `snapshot_date <= signal_date`.
- Factor files are partitioned by category and year, 2012–2026. Read only requested years and columns.
- Left-join factors onto the investable universe so missing deep-learning/high-frequency keys do not silently shrink it.
- Apply `availability_lag` on complete daily market keys before selecting weekly/monthly dates; `lag=1` remains one trading day.
- Returns and holdings use adjusted prices; actual traded amounts and board-lot interfaces use raw prices.
- `same_close` warns about possible look-ahead bias; `next_open` maps strictly to the next market trading day.
- `next_open` does not normally block opening orders using status only fully known after that day's close. Set `--set rules.use_eod_limit_for_open=true` for a conservative approximation.
- Drop signals that would execute after the requested end date so NAV does not extend beyond it.

The benchmark reconstructs a frictionless proxy from historical constituent weights and adjusted prices. It is not exchange-published official index NAV. `excess_nav` is the NAV ratio relative to this auditable proxy.

Data lacks announcement timestamps, historical ST status, a complete security master, and precise intraday price-limit paths. Existing files alone cannot establish strict point-in-time validity for fundamental/consensus factors. Set appropriate `availability_lag` values and add announcement-time data. Daily `up_down_limit_status` suits closing execution or an explicitly enabled conservative opening approximation; use `rules.method` for precise intraday tradability.

A security master is also needed for prolonged suspensions, delistings, and resumptions. Positions retain the last available closing price; audit extreme exits using `rejections.parquet` and add custom rules.

## 10. A-share rule sources and framework conventions

The standard profile has a replaceable historical stamp-duty table with boundaries at 2005-01-24, 2007-05-30, 2008-04-24, 2008-09-19, and 2023-08-28. Primary references recorded for these conventions:

- [Ministry of Finance: two-sided 0.1% from 2005-01-24](https://www.mof.gov.cn/gkml/caizhengwengao/caizhengbuwengao2005/caizhengbuwengao20053/200805/t20080525_42735.htm)
- [State Taxation Administration: two-sided 0.3% from 2007-05-30](https://www.chinatax.gov.cn/chinatax/n810341/n810765/n812176/200705/c1194505/content.html)
- [Ministry of Finance: two-sided 0.1% from 2008-04-24](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200805/t20080519_29133.htm)
- [Ministry of Finance: seller-only 0.1% from 2008-09-19](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm)
- [Tax administration: rate halved from 2023-08-28](https://zhejiang.chinatax.gov.cn/art/2023/8/28/art_8409_83672.html)
- [Shanghai Stock Exchange: purchased securities cannot be sold before settlement](https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml)

Exchange rules evolve. The framework prioritizes dataset suspension/price-limit status and exposes rules, costs, and board lots as settings or plugins rather than scattering board-specific thresholds through the loop.

## 11. Tests

```bash
uv run pytest

# Toy end-to-end tests independent of the large datasets
uv run pytest tests/test_end_to_end.py
```

Coverage includes:

- Weekly/monthly toy flows from Parquet to NAV.
- Strict backward as-of monthly index snapshots.
- Factor projection, Inf cleanup, multi-source left joins.
- IC, quantile returns, and next-open label alignment.
- Same-close/next-open mapping and weekend transitions.
- Suspensions, price limits, special statuses, missing prices.
- T+1, cash conservation, minimum commission, stamp-duty boundaries.
- Actual `minimal / standard / custom` behavior.
- Custom cost and tradability plugins.

## 12. All-A-share estimation and CSI 1000 multifactor application

Open the project in VS Code and choose **Terminal -> New Terminal**. Development defaults end at the validation boundary without automatically inspecting out-of-sample data.

These configurations require the local frozen list `configs/csi1000_selected_32.csv`. Like other research CSVs, it is excluded from Git; retrieve it from the research workspace or supply your own list through `--factor-file`.

```bash
# Maximize IC_IR: risk matrix from 12 completed periods of all-A-share RankIC
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic_ir

# Maximize IC: current all-A-share exposure correlation matrix
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic

# Weekly rebalancing with other settings unchanged
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic_ir \
  --rebalance weekly
```

Both methods produce nonnegative factor weights summing to 1. Optimize within each category, standardize category scores, and average categories equally by default, preventing larger price/volume categories from dominating. Every signal uses only historical IC with `exit_date <= signal_date`; unfinished holding periods cannot enter weights.

Winsorize, neutralize size/industry, and standardize the frozen 32 factors across all A-shares. Score only the signal date's historical CSI 1000 constituents. `portfolio.method` independently maps scores to weights:

```bash
# Moderate enhancement: tilt index weights (default)
uv run ashare-lab multi-factor --config configs/csi1000_multifactor.toml \
  --set portfolio.method=benchmark_tilt

# Equal-weight the top 100 stocks
uv run ashare-lab multi-factor --config configs/csi1000_multifactor.toml \
  --set portfolio.method=top_n_equal --set portfolio.top_n=100

# Renormalize index weights for the top 100
uv run ashare-lab multi-factor --config configs/csi1000_multifactor.toml \
  --set portfolio.method=top_n_benchmark --set portfolio.top_n=100
```

The standalone top-20% script preserves factor combination, execution, and costs while replacing stock weights. It runs max_ic_ir and max_ic by default:

```bash
uv run python run_top20_equal.py

# Run just one method
uv run python run_top20_equal.py --combination max_ic
```

This multifactor configuration executes next-open, blocks suspensions, and charges 0.1% on buys and sells without additional stamp duty or transfer fees. Override via `--execution`, `--rules`, or `--set costs.xxx=...`.

Reporting treats the period beginning on 2020-11-01 as continuous historical simulated-live evaluation. Run through the recorded dataset's latest date:

```bash
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic_ir \
  --end 2026-05-27 \
  --backtest-start 2020-11-01 \
  --output outputs/csi1000_sim_live_2020_latest
```

This is historical replay, not a live record saved daily at the time. Repeated strategy changes based on these results are historical backtest optimization.

Runs write NAV/drawdown charts, annual returns, IC history, weights, diagnostics, targets, trades, positions, rejections, and configuration under `multifactor/<method>/`.

For nonlinear regression or machine learning, set `multifactor.method` to `module:function`:

```python
def model(estimation_panel, application_panel, factor_specs, config):
    ...
    return scores  # Polars DataFrame with at least date, stock, score
```

`estimation_panel` contains all-A-share samples with `forward_return` and `exit_date`; `application_panel` contains CSI 1000 constituents. Custom models must filter `exit_date <= signal_date` before training for each signal. Stock weights, execution, costs, and reporting remain independent.

## 13. Research handoff notes

- [Multifactor project handoff](docs/multifactor_project_handoff.md): current design, recorded results, and next steps.
- [Legacy index-enhancement research handoff](docs/index_enhancement_research_handoff.md): earlier `quant` project notes and the selection-leakage correction. Its paths refer to the earlier workspace, not this repository.
