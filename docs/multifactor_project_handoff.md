# CSI 1000 Index-Enhancement Project Handoff

> Last updated: 2026-07-22
> Project directory: `/Users/Apple_501/Documents/bAcktest 2`
> This note lets a new conversation inherit the infrastructure, research conventions, and next-stage tasks.

This is a translation of the recorded research handoff. Dates and results below describe that research snapshot; they are not a fresh performance evaluation. Local datasets and research CSVs are excluded from Git.

## 1. Project objective and user preferences

Develop a long-only CSI 1000 index-enhancement strategy that can extend to other indices:

- Use all A-shares for factor research and model estimation.
- Apply final stock selection only to historical CSI 1000 constituents at each date.
- Focus on excess returns relative to CSI 1000 after `0.1%` costs on both buys and sells.
- Start with max_ic_ir and max_ic, then add nonlinear regression, machine learning, and other combination methods.
- Eventually deliver the strategy, an Excel workbook, NAV charts, and supporting tables. The strategy has not yet been frozen, so the final Excel deliverable has not been produced.

Explicit user preferences:

- Keep code concise, layered, and easy to modify.
- Make execution, trading rules, rebalancing, costs, and stock-weight methods configurable.
- Operate primarily through the VS Code terminal and TOML configuration.
- Keep toy tests compact and avoid excessive testing time.
- Support weekly and monthly rebalancing.
- Support Polars and DuckDB without changing commands or research conventions when switching backends.
- Stop distinguishing the 2020–2023 out-of-sample test from simulated-live evaluation after 2023. Describe the period starting on `2020-11-01` as continuous historical simulated-live evaluation.

## 2. Files to read first

For a new conversation, read in this order:

1. `docs/multifactor_project_handoff.md`: this note.
2. `configs/csi1000_multifactor.toml`: multifactor defaults.
3. `src/ashare_lab/multifactor.py`: rolling factor weights and combined scores.
4. `src/ashare_lab/multifactor_pipeline.py`: all-A-share estimation, CSI 1000 application, and outputs.
5. `src/ashare_lab/portfolio.py`: convert scores into stock weights.
6. `src/ashare_lab/pipeline.py`: research, backtesting, and public interfaces.
7. `src/ashare_lab/execution.py`, `execution_inputs.py`, `execution_engine.py`: execution.
8. `src/ashare_lab/settings.py`: defaults and validation.
9. Section 12 of `README.md`: VS Code command examples.
10. `tests/test_multifactor.py`: minimal multifactor convention tests.

`docs/index_enhancement_research_handoff.md` comes from an abandoned project. Use it only for algorithm design and output structure. Its earlier 39-candidate/15-core factor conclusions contain selection look-ahead leakage and cannot establish the current strategy's validity. Do not copy that factor list into this project.

## 3. Existing code layers

```text
Parquet / Excel / CSV
    |
data.py / duckdb_data.py       Scanning, date/stock filtering, joins, batch loading
    |
signals.py                    Winsorization, neutralization, standardization, availability lags
    |
selection.py / validation.py   Single-factor IC, coverage, stability, Lasso, sample splits
    |
multifactor.py                Rolling max_ic_ir / max_ic / custom models
    |
portfolio.py                  Scores to target stock weights
    |
execution*.py / costs.py      Next-open/same-close execution, suspensions, T+1, costs, trades
    |
metrics.py / charts.py        NAV, excess returns, drawdowns, metrics, charts
    |
multifactor_pipeline.py       Full-flow orchestration and results
    |
cli.py                        VS Code terminal commands
```

Each layer has one responsibility. Add methods as functions or plugins instead of accumulating conditionals in the execution loop.

## 4. Data and backends

Default data configuration:

```toml
[data]
engine = "duckdb"                 # Can be changed to polars
market = "a share/A股行情数据.parquet"
valuation = "a share/A股估值数据.parquet"
industry = "a share/A股行业分类.parquet"
index_members = "a share/指数成分股.parquet"
factor_root = "factor"
batch_size = 16
duckdb_threads = 4
```

These original filenames identify local data; see the English meanings in the README.

DuckDB handles:

- Parquet scanning.
- Date and stock filtering.
- Multi-table joins.
- Batched factor loading.
- Returning only the rows and columns needed for research to Polars.

The Polars backend remains available:

```bash
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --data-engine polars
```

Recorded full factor-selection comparison:

- Polars: approximately 37 minutes 24 seconds.
- DuckDB: approximately 19 minutes 38 seconds.
- Both selected the same 32 factors.

The recorded latest market date is `2026-05-27`. A later configured end date cannot extend a run beyond the available data.

## 5. Completed single-factor selection work

The two-stage selection framework consists of:

1. Single-factor IC, coverage, and stability screening.
2. Lasso to remove highly redundant or weak factors.

Fixed splits and rolling validation are supported in `settings.py` / `configs/csi1000.toml`:

```text
Training:                  2014-10-31 to 2019-04-30
Validation:                2019-05-01 to 2020-10-31
Original test definition:  2020-11-01 to 2023-10-31
Original simulated-live:   2023-11-01 to latest
```

The user later requested that multifactor reporting treat `2020-11-01` through the latest date as continuous historical simulated-live evaluation, without a separately labeled out-of-sample test. The earlier split interfaces remain for future research.

Frozen 32-factor list:

```text
configs/csi1000_selected_32.csv
```

This local research CSV contains:

- `factor`: factor identifier in the framework.
- `source`: factor category.
- `raw_name`: original Parquet column name.
- `direction`: `1` favors higher values; `-1` favors lower values.

The five categories are fundamental, low-frequency price/volume, high-frequency price/volume, consensus estimates, and deep learning. Read the CSV when factor details are needed rather than rebuilding the list manually.

Related historical output:

```text
outputs/all_factors_duckdb/factor_validation/selected_factors.csv
```

## 6. Multifactor preprocessing conventions

First process the full A-share cross-section:

1. Clean Inf and invalid values.
2. Winsorize at the 1st and 99th percentiles each period.
3. Neutralize industry exposure.
4. Neutralize market-cap exposure.
5. Apply z-score standardization.
6. Align directions using the frozen list's `direction`.

Only then select the signal date's historical CSI 1000 constituents and generate combined scores. This preserves the full-market cross-sectional interpretation instead of neutralizing within CSI 1000 alone.

Notes:

- Fundamental and consensus data lack complete announcement timestamps, so existing files alone cannot prove strict point-in-time validity.
- `availability_lag` supports per-factor availability lags.
- Historical CSI 1000 snapshots use backward as-of matching, never future constituents.
- `research.membership_lag` adds a constituent-snapshot lag.

## 7. Multifactor combination algorithms

Entry point: `src/ashare_lab/multifactor.py`.

### 7.1 Shared time conventions

- Monthly by default; weekly is supported.
- Factors are generated on signal dates.
- Default execution is the next trading day's open.
- Each period's factor IC uses adjusted returns from one rebalance execution price to the next.
- On a signal date, use only historical IC with `exit_date <= signal_date`.
- Unfinished holding periods cannot enter weight estimation.
- Use the most recent 12 completed IC periods by default.

Recorded real-data checks found zero uses of future IC.

### 7.2 Maximize IC_IR

Objective:

```text
maximize  w'μ / sqrt(w'Σw)
subject to w >= 0, sum(w) = 1
```

- `μ`: mean factor RankIC over the last 12 completed holding periods.
- `Σ`: covariance matrix of factor RankIC over those 12 periods.
- Solve with SLSQP.
- Symmetrize the matrix and apply an eigenvalue floor.

### 7.3 Maximize IC

Use the same objective form, replacing the risk matrix with the signal date's all-A-share exposure correlation matrix:

```text
maximize  w'μ / sqrt(w'Rw)
subject to w >= 0, sum(w) = 1
```

- `μ` still uses the last 12 completed periods.
- `R` is the current all-A-share factor-exposure correlation matrix.
- Shrink toward the identity matrix by 10% by default to reduce collinearity and numerical instability.

### 7.4 Factor categories

With default `group_by_source = true`:

1. Optimize separately within each category.
2. Generate each category's stock scores.
3. Standardize those scores again with z-scores.
4. Average categories equally.

Deep-learning or price/volume factors therefore do not dominate solely because they are more numerous.

### 7.5 Custom model interface

Add nonlinear regression or machine-learning methods through configuration:

```toml
[multifactor]
method = "your_module:your_function"
```

Signature:

```python
def model(estimation_panel, application_panel, factor_specs, config):
    ...
    return scores
```

Return a Polars DataFrame with at least:

```text
date, stock, score
```

- `estimation_panel`: all-A-share samples with `forward_return` and `exit_date`.
- `application_panel`: current CSI 1000 constituents and processed factors.
- Train custom models after filtering `exit_date <= signal_date` for each signal.
- Stock weights, execution, costs, and reports remain independent of the model.

## 8. Combined scores to stock weights

Entry point: `src/ashare_lab/portfolio.py`.

Built-in methods:

- `benchmark_tilt`: multiply index weights by `exp(tilt × score)`; default.
- `top_n_equal`: equally weight the N highest-scoring stocks.
- `top_n_benchmark`: select the top N and renormalize their original index weights.
- `module:function`: custom selection/weight builder.

Recorded defaults:

```toml
[portfolio]
method = "benchmark_tilt"
tilt = 0.35
score_clip = 3.0
max_weight = 0.02
```

The portfolio and multifactor-model layers are separate. Compare weight methods with a fixed model, or models with fixed stock weights.

## 9. Execution and costs

Multifactor defaults:

```toml
[schedule]
method = "monthly"

[execution]
price = "next_open"
initial_cash = 100000000.0
record_positions = true

[rules]
profile = "custom"
block_suspended = true
block_limit = false
use_eod_limit_for_open = false
t_plus_one = true

[costs]
commission_rate = 0.001
minimum_commission = 0.0
stamp_tax = "none"
transfer_fee = "none"
```

This means:

- Execute at the next day's open.
- Suspended stocks cannot trade.
- Do not block opening orders using end-of-day price-limit status.
- Charge 0.1% of traded value on each buy and sell.
- Do not add stamp duty or transfer fees separately.
- Use adjusted prices for returns and position valuation, raw prices for actual traded amounts.

Recorded cost check: `commission / gross_value` was exactly `0.001`.

## 10. Benchmark and excess-return definitions

Instead of an official CSI 1000 NAV series, use:

```text
Historical CSI 1000 constituents + historical weights + adjusted stock prices
```

This constructs a frictionless, cost-free constituent-weight proxy benchmark.

```text
excess_nav = strategy_nav / benchmark_nav
total_excess_return = excess_nav - 1
```

Annual excess returns also use geometric relative returns:

```text
(1 + annual_strategy_return) / (1 + annual_benchmark_return) - 1
```

This differs from simply subtracting benchmark returns from strategy returns. Before formal delivery, add official CSI 1000 market data for comparison.

## 11. Recorded continuous simulated-live results

Shared assumptions:

- Configured start: `2020-11-01`.
- First actual monthly signal: `2020-11-30`.
- End: `2026-05-27`.
- 32 factors.
- All-A-share estimation; CSI 1000 application.
- 12-period rolling window.
- Next-day opening execution.
- 0.1% on buys and sells.
- `benchmark_tilt` stock weights.

### 11.1 Annual excess returns

| Year | Maximize IC_IR | Maximize IC |
|---|---:|---:|
| November–December 2020 | -0.20% | -0.21% |
| 2021 | +1.56% | +1.96% |
| 2022 | +1.89% | +2.38% |
| 2023 | +0.91% | +0.71% |
| 2024 | +1.95% | +1.99% |
| 2025 | +1.07% | +1.11% |
| 2026 through May 27 | -1.05% | -1.17% |

### 11.2 Full-period metrics

| Metric | Maximize IC_IR | Maximize IC |
|---|---:|---:|
| Total strategy return | +46.21% | +47.14% |
| Total proxy benchmark return | +37.62% | +37.62% |
| Total relative excess return | +6.25% | +6.92% |
| Annualized excess return | +1.16% | +1.28% |
| Information ratio | 0.84 | 0.86 |
| Annualized tracking error | 1.24% | 1.34% |
| Maximum drawdown | -43.10% | -42.94% |
| Annualized one-way turnover | 116.19% | 115.31% |
| Total transaction costs (CNY 10,000) | 130.91 | 130.66 |

Interpretation:

- Both methods generated positive excess returns in each full year from 2021 through 2025.
- Maximize IC was slightly better overall, with a modest difference.
- Both had negative excess returns through May 2026, suggesting recent style deterioration needs investigation.
- The baseline methods produced about 1.2% annualized excess return; the portfolio strategy still needs improvement.

### 11.3 Top-20% equal-weight portfolio

`run_top20_equal.py` preserves factor weights and changes only the stock-weight layer:

1. Rank stocks by combined score each period.
2. Select the top 20% of historical CSI 1000 constituents, targeting 200 stocks per period.
3. Assign 0.5% target weight per stock.
4. Preserve suspension restrictions, next-open execution, and 0.1% costs on both sides.

```bash
uv run python run_top20_equal.py
```

Continuous historical simulated-live results:

| Metric | Maximize IC_IR | Maximize IC |
|---|---:|---:|
| Total strategy return | +70.82% | +59.38% |
| Total proxy benchmark return | +37.62% | +37.62% |
| Total relative excess return | +24.13% | +15.81% |
| Annualized excess return | +4.18% | +2.82% |
| Information ratio | 0.54 | 0.34 |
| Annualized tracking error | 7.17% | 7.48% |
| Maximum drawdown | -35.96% | -37.12% |
| Annualized one-way turnover | 690.97% | 648.17% |
| Total transaction costs (CNY 10,000) | 873.86 | 794.89 |

Annual excess returns:

| Year | Maximize IC_IR | Maximize IC |
|---|---:|---:|
| November–December 2020 | -2.47% | -2.54% |
| 2021 | +7.53% | +5.62% |
| 2022 | +7.95% | +9.53% |
| 2023 | +6.37% | +3.99% |
| 2024 | +7.44% | +4.89% |
| 2025 | +2.42% | +1.10% |
| 2026 through May 27 | -6.33% | -6.86% |

Top-20% equal weighting increased historical excess returns, with approximately 7% tracking error, annualized one-way turnover of 6.5–6.9 times, and roughly CNY 8.00–8.74 million in costs. It also magnified negative excess returns in 2026. Prioritize turnover, concentration, and active-risk constraints; higher cumulative excess returns alone do not establish superiority over a moderate index tilt.

Top-20% output directories:

```text
outputs/csi1000_top20_equal/multifactor/max_ic_ir/
outputs/csi1000_top20_equal/multifactor/max_ic/
```

Index-tilt output directories:

```text
outputs/csi1000_sim_live_2020_latest/multifactor/max_ic_ir/
outputs/csi1000_sim_live_2020_latest/multifactor/max_ic/
```

Each contains:

- `nav.csv`, `nav.svg`: strategy, benchmark, and excess NAV.
- `drawdown.svg`: drawdowns.
- `yearly_performance.csv`: annual returns and excess returns.
- `summary.json`: summary metrics.
- `factor_ic_history.parquet`: all-A-share factor IC by period.
- `factor_weights.csv`: factor weights by period and category.
- `weight_diagnostics.csv`: optimizer diagnostics.
- `signal_panel.parquet`: CSI 1000 scoring inputs.
- `targets.parquet`: target stock weights.
- `trades.parquet`: executions.
- `positions.parquet`: holdings.
- `rejections.parquet`: rejected orders and reasons.
- `strategy_daily.csv`, `benchmark_daily.csv`: underlying daily NAV.
- `resolved_config.json`: effective run configuration.

`outputs/csi1000_oos_test/` contains earlier 2020–2023 results under the old terminology. Use the continuous simulated-live directories going forward without separately labeling this interval out-of-sample.

## 12. Common VS Code commands

Open the folder in VS Code, then choose **Terminal -> New Terminal**:

```bash
cd "/Users/Apple_501/Documents/bAcktest 2"
```

Inspect effective settings:

```bash
uv run ashare-lab show-config \
  --config configs/csi1000_multifactor.toml
```

Run maximize IC_IR through the recorded latest date:

```bash
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic_ir \
  --end 2026-05-27 \
  --backtest-start 2020-11-01 \
  --output outputs/csi1000_sim_live_2020_latest
```

Run maximize IC:

```bash
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic \
  --end 2026-05-27 \
  --backtest-start 2020-11-01 \
  --output outputs/csi1000_sim_live_2020_latest
```

Switch to weekly:

```bash
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic \
  --rebalance weekly
```

Equal-weight the top 100:

```bash
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --set portfolio.method=top_n_equal \
  --set portfolio.top_n=100
```

Run all tests:

```bash
uv run pytest
```

Recorded test result: 58 passing tests. Recorded full multifactor runs took approximately:

- Maximize IC_IR: 31.65 seconds.
- Maximize IC: 38.84 seconds.

## 13. Existing test priorities

`tests/test_multifactor.py` verifies:

- `next_open` holding-period returns use the correct two execution prices.
- Only IC from completed periods enters estimation.
- Optimized weights are nonnegative and sum to 1.

Other coverage includes:

- Polars/DuckDB consistency.
- Weekly/monthly toy end-to-end flows.
- Backward as-of historical constituents.
- Factor reading and multi-source left joins.
- Size and industry neutralization.
- Same-close/next-open mapping.
- Suspensions, price limits, T+1, cash conservation.
- Commission, stamp-duty, and transfer-fee boundaries.
- NAV charts and plugin interfaces.

## 14. Suggested next stage

Improve excess returns in the multifactor and stock-weight layers using the existing data and execution infrastructure. Suggested order:

### 14.1 Diagnose existing methods

Inspect saved outputs for:

- Factor and category weights each period.
- Excessive weight concentration.
- Categories, industries, and stocks responsible for negative excess returns in 2026.
- The relationship between excess returns and turnover.
- Correlation between stock scores from maximize IC and IC_IR.
- Active industry, size, and style exposures.
- Return and transaction-cost contributions.

### 14.2 Improve stock portfolio optimization

`benchmark_tilt` is simple but not a full index-enhancement optimizer. Add an independent portfolio plugin considering:

- Maximizing expected stock returns.
- Tracking-error constraints.
- Active industry-weight constraints.
- Size/style exposure constraints.
- Stock-level deviations from benchmark weights.
- Turnover penalties or caps.
- Individual stock-weight caps.
- Fully invested, long-only portfolios within constituents.

These belong in `portfolio.py` or a new portfolio-optimization module rather than `multifactor.py`.

### 14.3 Extend factor-combination models

Add methods through the same interface:

- Exponentially weighted or Bayesian-shrinkage IC.
- IC_IR with weight caps.
- Ridge / Elastic Net.
- Cross-sectional linear regression.
- GBDT, XGBoost, LightGBM.
- Nonlinear interaction models.
- Classification or ranking models.

Every model must preserve rolling training and the `exit_date <= signal_date` constraint.

### 14.4 Final deliverables

Create the Excel workbook after freezing the strategy, including at least:

- Configuration and conventions.
- Strategy summary.
- Annual performance.
- NAV and drawdowns.
- Monthly excess returns.
- Factor weights.
- Industry and style exposures.
- Holdings and turnover.
- Transaction costs.
- Trade and rejection audit.

## 15. Known limitations and interpretation

1. The benchmark is constituent-based proxy NAV, not official index NAV.
2. Announcement timestamps are incomplete; strict fundamental-factor PIT validity needs additional data or conservative lags.
3. Historical ST, delisting/listing status, and precise intraday price-limit paths are incomplete.
4. `next_open` does not block opening orders using status fully known only after the close.
5. Simulated-live evaluation is historical replay, not a record saved daily at the time.
6. Results from 2020 onward have already been inspected. Repeated adjustments based on them are historical backtest optimization, not evaluation on untouched data.
7. Add strategies via `multifactor.method`, `portfolio.method`, or new modules while preserving existing features.
8. Preserve raw data and existing outputs; use a new `--output` directory for each new experiment.

## 16. Direct task description for a new conversation

> Continue developing extensible multifactor and stock-portfolio optimization strategies using the completed all-A-share preprocessing, 32-factor list, max_ic_ir / max_ic models, historical CSI 1000 constituents, A-share execution, and reporting framework. First diagnose the source of the roughly 1.2% annualized excess return and the 2026 deterioration. Then add an index-enhancement weight optimizer with industry, style, stock-level deviation, and turnover constraints. Preserve existing commands and features, add methods through interfaces, and train all models chronologically without future data.
