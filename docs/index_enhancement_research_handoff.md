# Index-Enhancement Research: Work Handoff and Next Steps

> Last updated: 2026-07-17
> Original workspace: `/Volumes/Macintosh HD/Users/Apple_501/Documents/quant`
> This note restores the week's research context in a new conversation. It covers index-enhancement research only, excluding resume work and general framework tutorials for friends.

This is an archived handoff from the earlier `quant` project, translated into English. Source paths and results refer to that workspace, not the current AShare Lab repository.

> **Important correction dated 2026-07-17:** The earlier 39-candidate/15-core factor list used 2023–2026 results for selection and then replayed the strategy backward. This creates selection-stage look-ahead leakage and invalidates those results as validation evidence. The valid entry points changed to `build_temporal_factor_selection.py` and `run_strict_temporal_multifactor.py`: all selection is confined to 2018–2022, and data from 2023 onward is used only for retrospective evaluation. Monthly and weekly rebalancing are supported. See `docs/strict_temporal_multifactor.md` in the original workspace for conventions and results. The earlier workflow below is retained for historical traceability and is not the current recommended approach.

## 1. Research objective and final deliverables

Use the newly supplied historical index-constituent data and five factor categories to build an index-enhancement stock-selection strategy. Selection may occur within index constituents or across the entire market; factor types, rebalancing, and portfolio construction are configurable.

The final objective is stable excess returns relative to the benchmark after one-way transaction costs of 0.1% (10 bp).

Deliver:

1. A clear, reproducible strategy.
2. A final Excel workbook explaining the stock universe, frequency, factors, weights, execution, costs, benchmark, NAV, excess returns, risk, turnover, period holdings, target weights, trades, annual/segment performance, and data/PIT assumptions.

### Enhancement within index constituents

Use an index's historical constituents as the universe. For example, buy a subset of the highest-scoring CSI 1000 constituents each period and compare against a benchmark replicating that period's constituent weights.

The aim is active allocation among stocks while broadly retaining the index's risk profile. Start here because the benchmark, universe, and exposures are easier to explain and constrain.

## 2. Research route recorded at the time

Begin with a simple baseline before building a complex optimizer:

```text
Data and timing acceptance
    | completed
Unified monthly single-factor evaluation
    | completed
Candidate screening and correlation deduplication
    | completed
Simple, transparent multifactor baseline
    | next step in the earlier plan
Compare factor weights, holding counts, and portfolio construction
    |
Add risk, industry, turnover, and liquidity constraints
    |
Rolling validation and final Excel workbook
```

The first baseline proposed in the earlier workflow:

- Universe: historical CSI 1000 constituents first, then the other four indices or all stocks.
- Frequency: monthly, matching the data foundation, evaluation, and timing contract.
- Factors: begin with 15 A-class core candidates rather than equally weighting all 39 candidates.
- Combination: average within categories, then weight categories equally so numerous price/volume factors do not dominate.
- Portfolio: test Top 100 / Top 150 with equal weights, then compare index-weight tilts.
- Costs: 10 bp on both buy and sell value.
- Benchmark: replicate the corresponding index's historical constituents and weights.

This baseline is a reference for later improvements, not a predetermined final strategy. The correction above supersedes the earlier selection procedure.

## 3. Recorded dataset overview

Original Chinese filenames below identify local datasets.

### 3.1 A-share market data

File: `A股行情数据.parquet` (A-share market data).

| Item | Recorded result |
|---|---:|
| Rows | 16,026,470 |
| Columns | 20 |
| Stocks | 5,823 |
| Trading days | 5,195 |
| Date range | 2004-12-31 to 2026-05-27 |
| File size | Approximately 1.07 GiB |

Common fields include stock code, trading date, OHLC, volume/value, adjusted prices, adjustment factors, and trading status. Market data supports tradability checks and stock-return calculations.

### 3.2 Historical index constituents

File: `指数成分股.parquet` (index constituents).

| Item | Recorded result |
|---|---:|
| Rows | 2,273,651 |
| Columns | 5 |
| Indices | 5 |
| Covered stocks | 4,362 |
| Raw date range | 2005-04-08 to 2026-07-01 |

Original fields mapped to canonical names:

| Original field | Standard field | Meaning |
|---|---|---|
| `s_info_windcode` | `index_code` | Index code |
| `s_con_windcode` | `code` | Constituent stock code |
| `trade_dt` | `membership_date` | Constituent snapshot date |
| `i_weight` | `source_weight` | Original constituent weight |
| `__index_level_0__` | Ignored | Index column with no research meaning |

The five indices:

- `000300.SH`: CSI 300.
- `000852.SH`: CSI 1000.
- `000905.SH`: CSI 500.
- `000906.SH`: CSI 800.
- `399102.SZ`: ChiNext Composite.

Zero weights and abnormal stock codes were cleaned with audit records retained. Renormalize remaining weights for each index and historical snapshot.

### 3.3 Five factor categories

Each category has nine annual Parquet files for 2018–2026, totaling 45 factor-source files.

| Category code | Category | Factor count | Annual filename pattern |
|---|---|---:|---|
| `low_frequency_price_volume` | Low-frequency price/volume | 555 | `daily_tech_factor_*.parquet` |
| `high_frequency_price_volume` | High-frequency price/volume | 426 | `daily_minute1_tech_factor_*.parquet` |
| `fundamental` | Fundamental | 105 | `daily_fundamental_factor_*.parquet` |
| `consensus` | Consensus estimates | 42 | `daily_consensus_factor_*.parquet` |
| `deep_learning` | Deep learning | 24 | `daily_deep_learning_factor_*.parquet` |
| Total | | 1,152 | |

The five annual factor categories plus market data and index constituents comprise 47 files, approximately 32.29 GiB. The valuation file is discussed separately below. Factor tables are mostly date-by-stock panels with different columns; their rows are not entirely independent market samples.

### 3.4 A-share valuation data: used and excluded components

File: `A股估值数据.parquet` (A-share valuation data).

| Item | Recorded result |
|---|---:|
| Rows | 24,369,473 |
| Columns | 17 |
| Common fields | `s_val_mv`, `s_dq_mv`, PB, PE, PCF, PS, turnover, free-float shares, etc. |

This file was not included in the registry of 1,152 new factors, whose scope was the five newly supplied annual categories, constituents, and market data.

It was nevertheless used: unified single-factor evaluation and candidate correlation calculations read `s_dq_mv` by stock on each signal date for cross-sectional neutralization against log(circulating market capitalization). Coverage in the five-index monthly universe was 362,698 / 362,700, approximately 99.9994%. The field dictionary identifies `s_dq_mv` as daily circulating market capitalization, not total market capitalization. Changing to `s_val_mv` requires a separate experiment.

Valuation columns such as `s_val_pb_new`, `s_val_pe_ttm`, `s_val_ps_ttm`, PCF, and turnover had not yet been added as candidate factors alongside the 1,152 new factors. This was an intentional scope restriction, not a technical inability to use them.

An earlier CSI 1000 single-factor compatibility run used `s_val_pb_new`, with output in `results/single_factor/s_val_pb_new/`. It showed the valuation table can serve as a wide factor source. Its results cannot be directly combined with the unified five-index batch without rerunning under identical historical constituents, monthly timing, weight methods, and splits.

The earlier recommendation was to start with the 15 core factors, then evaluate PB, PE, PS, PCF, and turnover separately under the same timing and costs before adding stable, low-correlation valuation factors. Adding them directly would make selection inconsistent with the 1,107-factor batch. This recommendation remains subject to the temporal-selection correction at the top of this note.

#### Fundamental-file update review dated 2026-07-15

Rebuilt the data foundation in isolation under `results/data_foundation_updated_check/`, preserving the official `results/data_foundation/` outputs.

- Nine fundamental files now totaled 9,522,146 rows versus 9,483,400 previously; all 38,746 added rows came from 2026.
- The 2026 file's maximum date extended from 2026-07-03 to 2026-07-14.
- The 2018–2025 files matched earlier fingerprints; only the 2026 file's rows, size, and Parquet fingerprint changed.
- All 105 fundamental factors had no explicit `NaN`, `+Inf`, or `-Inf`; the minimum source-wide finite-value coverage was about 82.74%.
- Cross-sectional admission remained 102 passing and 3 failing factors: `deducted_netprofit_yoy_qr_std/sud/svr`. Valid signal proportions were about 86.06%, below the 90% threshold.
- Overall quality remained 35 `PASS`, 7 `WARN`, 0 `FAIL`, with 1,107 qualified factors.

The research window ended on 2026-05-27, so later additions did not enter the research panel. This appeared to extend data rather than alter values already evaluated. If historical values inside the research window were corrected, rerun the official foundation, unified batch, and candidate pool instead of retaining earlier results.

## 4. Completed infrastructure

### 4.1 Data adapters and factor registry

Core files in the original workspace:

- `src/quant_research/research_data.py`
- `src/quant_research/data_foundation.py`
- `scripts/build_data_foundation.py`

Implemented:

- Automatic recognition of five factor directories and annual partitions for 2018–2026.
- Cross-year schema checks within each category.
- Canonical dates, stock/index codes, and weights.
- A static registry of 1,152 factors.
- Category-specific column projection without materializing a huge five-indices-by-dates-by-1,152-factors table.
- DuckDB Parquet scanning, joins, and aggregation.
- Preservation of universe rows when factors are missing.
- Input size, modification time, schema, and Parquet-footer fingerprints for reproducibility.

Registry: `results/data_foundation/factor_registry.csv`.

### 4.2 Unified monthly timing contract

All five indices use one research calendar:

```text
membership_date <= signal_date = factor_date < entry_date
next_signal_date < exit_date <= 2026-05-27
```

1. `signal_date`: the month's last market trading day.
2. Constituents: the latest snapshot no later than the signal date.
3. Factors: exact signal-date values, without forward/backward filling.
4. `entry_date`: the first market trading day after the signal, entering at the close.
5. `exit_date`: the first market trading day after the next signal, exiting/rebalancing at the close.
6. Drop trailing periods without complete exit prices.
7. The shared market cutoff is 2026-05-27; later factors and snapshots do not enter the panel.

There were 99 complete monthly holding periods:

- First: signal 2018-01-31, entry 2018-02-01, exit 2018-03-01.
- Last: signal 2026-03-31, entry 2026-04-01, exit 2026-05-06.

Five-index monthly universe: `results/data_foundation/monthly_index_universe.parquet`, 362,700 rows.

### 4.3 Data quality acceptance

Of 42 checks: 35 `PASS`, 7 `WARN`, 0 `FAIL`.

Confirmed:

- All 45 annual files were present, containing 1,152 factors.
- No additional duplicate date/stock primary keys.
- Cleaned constituent weights could be normalized.
- No future constituent snapshots or factors were used.
- Left joins neither increased row counts nor removed universe rows.
- Research stopped at the market-data cutoff.
- Mean factor-to-universe key match rate was about 99.87%.

Warnings and treatment:

| Warning | Extent | Treatment |
|---|---:|---|
| Non-finite values | 5 factors, 30 values | Convert to missing before preprocessing |
| Raw finite coverage below 80% | 33 factors | Further index/monthly cross-sectional screening |
| Failed index cross-section admission | 45 factors | Exclude from unified evaluation |
| PIT metadata unverified | 1,152 factors | Assume `trade_dt` values available after close; confirm using factor dictionary |
| Abnormal/zero-weight constituent records | 712 rows | Clean and retain audit records |
| Abnormal historical market codes | 433 rows | Isolate using standard-code anchors in the research window |
| Missing constituent endpoint prices | 2 entry rows, 38 exit rows | Record missing-return weights; maximum exit weight about 0.493% |

Quality files:

- `results/data_foundation/data_quality_checks.csv`
- `results/data_foundation/factor_quality_on_universe_summary.csv`
- `results/data_foundation/join_quality_summary.csv`
- `results/data_foundation/monthly_periods.csv`
- `docs/data_foundation.md`

## 5. Completed unified single-factor batch

### 5.1 Evaluation conventions

Entry: `scripts/run_single_factor_evaluation.py`.

- Universe: historical constituents of five indices.
- Eligible factors: 1,107 out of 1,152.
- Rebalancing: month-end signals, next trading-day closing execution.
- Preprocessing: non-finite -> missing, MAD winsorization at 3 times MAD, log(circulating market-cap) neutralization using `s_dq_mv`, then z-score.
- Quantiles: five groups independently within each index.
- Portfolio: top 20% in the frozen favorable direction.
- Weights: `benchmark`, retaining selected stocks' relative benchmark weights and renormalizing.
- Costs: 10 bp on buy value and 10 bp on sell value.
- Benchmark: historical constituent weights and adjusted stock returns under the same conventions.

Call it a historical constituent-weight replication benchmark or `synthetic constituent-weight benchmark`, not official index points or a total-return index.

### 5.2 Sample splits intended to control overfitting

| Segment | Dates | Months | Purpose |
|---|---|---:|---|
| Development | 2018-01 to 2022-12 | 60 | Freeze factor directions only |
| Validation | 2023-01 to 2024-12 | 24 | Main performance screening |
| Subsequent stability | 2025-01 to 2026-03 | 15 | Check recent predictive power and excess returns |

Each factor's single direction came from the sign of its development-period mean RankIC pooled across five indices:

- Positive: `high`, higher values favored.
- Negative: `low`, lower values favored.

Directions could not change during validation or stability evaluation. The frozen count was 389 `high` and 718 `low` factors.

### 5.3 Completed batch run

| Item | Result |
|---|---:|
| Requested factors | 1,107 |
| Successful factors | 1,107 |
| Failed factors | 0 |
| Factor-by-index result rows | 5,535 |
| Indices | 5 |
| Total runtime | 629.8 seconds, about 10 minutes 30 seconds |
| Shard saving | Every 10 factors |

Output: `results/single_factor_unified/`.

| File | Purpose |
|---|---|
| `batch_summary.csv/.parquet` | 91 metrics per factor/index for full, development, validation, and stability periods |
| `factor_directions.csv` | Frozen directions and BH-FDR information for 1,107 factors |
| `ic_series.parquet` | Monthly IC, RankIC, coverage, preprocessing diagnostics |
| `group_returns.parquet` | Monthly five-quantile and direction-aligned long/short returns |
| `portfolio_periods.parquet` | Buy/sell value, costs, portfolio/benchmark/excess returns and NAV |
| `checkpoint.json`, `shards/` | Progress and resumable shards |
| `batch_manifest.json` | Parameters, timing, costs, benchmark definition |

Full explanation: `docs/single_factor_framework.md`.

## 6. Earlier candidate-factor pool

Entry: `scripts/build_factor_candidate_pool.py`. This section describes the earlier screening later corrected for temporal leakage.

### 6.1 Selection rationale

Consider jointly:

- Data quality and valid cross-section proportions.
- Validation RankIC.
- Positive validation coverage across five indices.
- Validation excess returns after costs.
- RankIC and excess-return stability after 2025.
- Turnover.
- Factor coverage.
- Cross-sectional correlation with already selected factors.

Initial thresholds:

- Validation mean RankIC at least 0.01.
- Positive validation RankIC in at least 4 indices.
- Nonnegative mean validation excess return after costs.
- Positive validation excess return in at least 3 indices.
- Stability-period RankIC and excess returns meeting A/B class rules.
- All candidates must pass data-quality requirements.

### 6.2 Correlation deduplication

Reload actual cross-sectional scores for initial candidates:

1. Align factor directions.
2. Apply MAD winsorization.
3. Neutralize log(circulating market capitalization) using `s_dq_mv`.
4. Apply z-scores.
5. Pool five indices' month-end cross-sections.
6. Compute Pearson correlations.
7. Greedily retain higher-scoring factors with absolute correlation `< 0.70` to already retained factors.

### 6.3 Recorded results

| Stage | Factors |
|---|---:|
| Passed data quality | 1,107 |
| Before correlation deduplication | 48 |
| A-class stable core among them | 21 |
| B-class watchlist among them | 27 |
| Removed for correlation | 9 |
| Final candidate pool | 39 |
| Final A-class stable core | 15 |
| Final B-class watchlist | 24 |

Category distribution of 39 candidates:

| Category | Count |
|---|---:|
| Low-frequency price/volume | 16 |
| High-frequency price/volume | 14 |
| Deep learning | 3 |
| Consensus estimates | 3 |
| Fundamental | 3 |

Maximum recorded absolute correlation among selected factors was 0.6955, satisfying the threshold.

The top A-class factor, `rank_mom_20d_250d`, had these five-index pooled results:

- Validation mean RankIC: about 0.0837.
- Validation mean annualized excess return after costs: about 3.62%.
- Mean RankIC after 2025: about 0.0332.
- Mean annualized excess return after costs after 2025: about 3.72%.
- Mean one-way turnover: about 26.3%.

This is an example of a relatively stable single factor, not final multifactor strategy performance.

Candidate files:

- `results/factor_candidate_pool/candidate_pool.csv`
- `results/factor_candidate_pool/pre_correlation_shortlist.csv`
- `results/factor_candidate_pool/factor_correlation.parquet`
- `results/factor_candidate_pool/redundancy_log.csv`
- `results/factor_candidate_pool/candidate_pool_manifest.json`

Interim analysis workbook: `outputs/factor_candidate_pool_20260713/因子候选池.xlsx` (factor candidate pool).

It contains Summary, Core Candidate Pool, All Factor Scores, Correlation Matrix, Deduplication Log, Candidate Index Details, Selection Parameters, Checks, and Sources/Conventions. It explains screening and is not the final strategy workbook.

## 7. Multifactor combination: the earlier next-step plan

### 7.1 Definition

```text
Raw factors
    | align directions, winsorize, neutralize, standardize
Multidimensional factor scores for each stock
    | factor/category weighting
Combined stock score
    | selection and weight allocation
Index-enhancement portfolio
```

```text
combined_score = sum(factor_weight * standardized_direction_aligned_factor_score)
```

Combine value, quality, momentum, price/volume, liquidity, and other information to reduce reliance on a single factor.

### 7.2 Proposed first baseline

```text
Universe: historical CSI 1000 constituents
Frequency: monthly
Factors: 15 A-class stable core factors
Preprocessing: existing directions, MAD winsorization, size neutralization
Combination: within-category averages, then equal category weights
Holdings: top 100 by combined score, equal weights
Costs: 10 bp on buys + 10 bp on sells
Benchmark: historical CSI 1000 constituent-weight replication
```

Roughly 30 of the 39 factors were price/volume factors. Equal factor weighting would naturally overpower fundamental, consensus, and deep-learning information; category balancing is easier to interpret.

### 7.3 Strategy comparisons

After establishing a reproducible baseline, vary one dimension at a time:

| Dimension | Candidates |
|---|---|
| Factor set | 15 A-class; 15 A-class plus a few B-class; category representatives |
| Factor weights | Equal category weights; rolling RankIC/ICIR; smoothing and caps |
| Holding count | Top 80, 100, 150, 200 |
| Stock weights | Equal; score-weighted; index tilt |
| Frequency | Monthly baseline; weekly cost sensitivity |
| Universe | Index constituents first; all-market extension |

### 7.4 Portfolio construction sequence

1. Top N equal weights: simplest baseline, but potentially different industry/size exposures.
2. Index-weight tilts: overweight high scores and underweight low scores relative to benchmark weights.
3. Constrained optimization: treat scores as expected returns and impose stock, industry, style, turnover, and liquidity limits. Add this last to limit overfitting from many parameters.

Index-tilt intuition:

```text
target_weight ∝ benchmark_weight * exp(λ * combined_score)
```

Then apply weight caps, industry-deviation constraints, and normalization.

### 7.5 Appropriate sample use

Data from 2025 onward already contributed to candidate stability screening and is therefore not an untouched final out-of-sample interval.

The earlier multifactor plan proposed either:

1. Rolling training/validation, such as estimating weights on the previous 36 months and holding in the next period.
2. A separately reserved final test interval, inspected minimally and evaluated once.

Repeated full-sample tuning of Top N, weights, and constraints cannot turn the same interval's best result into valid out-of-sample evidence. See the temporal-selection correction above.

## 8. Final strategy evaluation

Present interpretable results:

- Annualized excess return after costs.
- Information ratio.
- Relative NAV and maximum excess drawdown.
- Monthly excess-return win rate.
- Annual excess returns and stability.
- Annualized turnover, buy/sell costs, and total costs.
- Industry, size, and individual stock deviations where data permits.
- Holding counts, stocks, and target weights each period.
- Parameter sensitivity.

The earlier single-factor NAV uses returns between monthly entry and rebalance endpoints. Its `max_drawdown` is a monthly holding-period-end drawdown, not daily mark-to-market drawdown within the month. Add daily position valuation after fixing holdings if the final deliverable emphasizes daily risk control.

## 9. Suggested final Excel structure

| Worksheet | Contents |
|---|---|
| Summary | One-sentence strategy, key metrics, NAV chart, conclusions |
| Strategy Parameters | Universe, frequency, Top N, factors, weights, costs, dates, benchmark |
| Factor Construction | Categories, directions, preprocessing, category weights, dynamic weighting |
| Performance Overview | Annualized return/excess, IR, drawdown, win rate, turnover, costs |
| NAV and Drawdowns | Strategy, replication benchmark, relative NAV, drawdown series |
| Annual Performance | Annual strategy/benchmark/excess returns, IR, turnover |
| Holdings | Signal/execution dates, stocks, target weights, scores, benchmark weights |
| Rebalancing and Costs | Buy/sell value, one-way turnover, buy/sell fees, total costs |
| Robustness | Top N, weighting, frequency, other parameter comparisons |
| Conventions and Limits | Historical constituents, benchmark, PIT assumptions, missing returns, limitations |

## 10. Known limitations and required wording

1. Official benchmark series are absent for the five indices. Clearly identify the historical constituent-weight replication benchmark.
2. PIT availability is unverified. Factor `trade_dt` is assumed available after that day's close; fundamental/consensus factors particularly need announcement or availability dictionaries.
3. Historical industry semantics are not fully verified. Added daily snapshots exactly cover 99,000 / 99,000 observations across 99 CSI 1000 signal periods, using `industry_source_date = signal_date < entry_date` without filling. However, classification system, version, provider, and historical release metadata are absent. Snapshot structure alone cannot prove contemporaneous public availability.
4. Missing endpoint returns use conservative `total_loss` treatment and report missing weights. Test sensitivity before drawing formal conclusions.
5. Drawdowns do not include within-month daily mark-to-market paths.
6. A 39-factor candidate pool and 15 core factors are not the final model; select the final combination through portfolio-level rolling validation.
7. Single-factor results are not multifactor performance; correlations, weights, turnover, and risk limits require a new backtest.

## 11. Commands and file map

These commands belong to the original workspace. Use `.venv/bin/python` from its project root.

### Rebuild the data foundation

```bash
.venv/bin/python scripts/build_data_foundation.py
```

Output: `results/data_foundation/`.

### Rerun all unified monthly single-factor evaluations

```bash
.venv/bin/python scripts/run_single_factor_evaluation.py \
  --unified-monthly \
  --all-factors \
  --factor-batch-size 10 \
  --weighting benchmark \
  --neutralize-size
```

Output: `results/single_factor_unified/`.

### Rebuild the candidate pool

```bash
.venv/bin/python scripts/build_factor_candidate_pool.py
```

Output: `results/factor_candidate_pool/`.

### Tests

```bash
.venv/bin/python -m pytest -q
```

Recorded result as of 2026-07-14: 37 passed.

After industry neutralization was added on 2026-07-16, only new and directly related tests ran: 20 passed. The data foundation and 1,107-factor batch were not rerun. All 49 independent programmatic audits of saved backtests also passed.

### Key files

| File | Purpose |
|---|---|
| `src/quant_research/research_data.py` | Five-category factor, constituent, and market adapters |
| `src/quant_research/data_foundation.py` | Data contracts and quality foundation |
| `src/quant_research/unified_single_factor.py` | Unified monthly factor batch |
| `src/quant_research/factor_candidates.py` | Screening, scoring, correlation deduplication |
| `src/quant_research/multi_factor.py` | 15-factor category weighting, Top 100, position accounting |
| `src/quant_research/ic_optimized_combination.py` | Maximize IC_IR and IC |
| `src/quant_research/industry_neutral_combination.py` | Exact signal-date industry joins, industry/circulating-cap neutralization, direct 15-factor equal weighting |
| `src/quant_research/backtester.py` | Existing general backtest core |
| `src/quant_research/scheduled_weights.py` | External target-weight backtests and multifactor integration |
| `scripts/run_single_factor_evaluation.py` | Single-factor entry |
| `scripts/build_factor_candidate_pool.py` | Candidate pool entry |
| `scripts/run_industry_neutral_factor_combination.py` | Fixed five-method industry-neutral backtests |
| `scripts/audit_industry_neutral_factor_combination.py` | Independent IC, weight, position, cost, benchmark, NAV reconciliation |
| `docs/data_foundation.md` | Foundation details |
| `docs/single_factor_framework.md` | Factor conventions and outputs |
| `docs/industry_neutral_factor_combination.md` | Industry neutralization, equal-weight comparisons, results, limitations |

## 12. Earlier suggested opening prompt

The following prompt is retained as a historical artifact and is superseded by the correction at the beginning of this note:

```text
Read docs/index_enhancement_research_handoff.md and continue from the completed
data foundation, 1,107-factor batch results, and 39-factor candidate pool.

Do not rerun data acceptance or single-factor batches. First implement and verify
a monthly multifactor baseline within historical CSI 1000 constituents:
15 A-class factors combined with equal category weights, Top 100 equal stock weights,
one-way 10 bp costs, and the historical constituent-weight replication benchmark.

Present an implementation plan and reusable/new interfaces first; after confirmation,
write code and run the backtest. Preserve PIT conventions and distinguish development,
validation, and final testing. Data after 2025 already used in candidate screening
must not be described as completely untouched out-of-sample data.
```

## 13. Recorded status

The week's data acceptance, unified preparation, individual evaluation, and candidate screening were completed:

- Unified data access.
- Fixed monthly timing contract.
- 1,107 qualified factors evaluated over five indices and 99 monthly periods.
- 39 candidates and 15 A-class core factors.
- Correlation deduplication, quality reports, shards, parameter manifests, and tests.

The simple multifactor baseline and industry-neutral extension were also completed: frozen 15 factors, equal category weights, direct factor equal weighting, maximize IC_IR, and maximize IC were backtested and reconciled over the same 86-period window. Further sensitivity plans, such as all-A-share industry estimation, total-market-cap treatment, or weight smoothing, should be frozen in advance. Do not retune Top N, factors, or weights based on this run and present the improvements as out-of-sample evidence. The temporal-leakage correction at the top governs interpretation of the earlier results.
