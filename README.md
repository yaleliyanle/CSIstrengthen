# AShare Lab

一个面向中证 1000 指数增强研究的 A 股多因子框架。它重点解决三件事：

1. 只读取 51GB 数据里本次真正需要的年份、股票和列；
2. 把因子清洗、合成、选股、调仓、成交、交易规则和费用拆开；
3. 常用变化直接在命令行选择，特殊变化用一个小函数插件实现。

默认使用 Polars；也可以切换到 DuckDB，让它负责 Parquet 扫描、日期/股票
过滤、连接和因子分批读取。两种后端共用同一套研究与回测逻辑。

## 1. 安装与第一次检查

项目建议用 `uv`，在项目根目录运行：

```bash
uv sync --group dev
uv run ashare-lab --help
uv run ashare-lab data-info --config configs/csi1000.toml
```

以后所有命令都可以保持 `uv run ashare-lab ...` 的写法，不必手动激活虚拟环境。

### 在 Polars 和 DuckDB 之间切换

默认值写在 `configs/csi1000.toml`：

```toml
[data]
engine = "polars"       # 改为 duckdb 即可切换
batch_size = 16         # DuckDB 每批读取多少个因子
duckdb_threads = 4
```

也可以在 VS Code 终端临时覆盖，不用修改 TOML：

```bash
# 使用原来的 Polars 后端
uv run ashare-lab validate-factors --config configs/csi1000.toml \
  --data-engine polars

# 使用 DuckDB 后端；其他参数和输出不变
uv run ashare-lab validate-factors --config configs/csi1000.toml \
  --data-engine duckdb
```

`--data-engine` 对 `single-factor`、`select-factors`、`validate-factors` 和
`backtest` 都有效。DuckDB 只替换数据访问层，不改变中性化、IC、Lasso、
交易规则或净值计算。

先看看可用因子：

```bash
# 全部 1152 个因子
uv run ashare-lab list-factors --config configs/csi1000.toml

# 只看基本面因子
uv run ashare-lab list-factors --config configs/csi1000.toml --source 基本面因子

# 搜索名称里含 hist_vol 的因子
uv run ashare-lab list-factors --config configs/csi1000.toml --contains hist_vol
```

## 2. 最常用的回测命令

### 周频、次日开盘、标准 A 股规则

```bash
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rebalance weekly \
  --execution next_open \
  --rules standard
```

效果：每周最后一个真实交易日产生目标权重，下一交易日开盘成交；应用停牌、T+1 和配置中的交易费用。日线数据无法知道开盘瞬间是否封板，因此默认不用当日收盘涨跌停字段拦截开盘单。

### 月频、当日收盘、极简规则

```bash
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rebalance monthly \
  --execution same_close \
  --rules minimal \
  --set execution.allow_same_close=true
```

效果：每月最后一个真实交易日按收盘价成交；只要求成交价格存在，不拦截停牌/涨跌停，不启用 T+1，交易费用为 0。

`same_close` 只适合信号在收盘成交前已经可得的假设。若因子使用了当日完整收盘数据，这种口径可能前视；框架默认会提示，显式设置 `allow_same_close=true` 后静默运行。

### 月频、次日开盘、只用一个因子

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

`--factor` 格式是：

```text
来源目录:因子名:方向:合成权重
```

- 方向 `1`：因子越大越好；
- 方向 `-1`：因子越小越好；
- 方向和权重不写时都默认为 `1`；
- 因子名含 `+` 时也只需把整段放在引号内。

### 命令行临时合成多个因子

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

每个因子先单独做截面清洗和标准化，再按权重合成，避免不同量纲直接相加。

## 3. 单因子检测

```bash
uv run ashare-lab single-factor \
  --config configs/csi1000.toml \
  --factor '基本面因子:bp_lf:1:1' \
  --rebalance monthly \
  --execution next_open \
  --set analytics.horizons='[1,5,20]' \
  --output outputs/bp_test
```

会输出：

- 因子覆盖率；
- 每期 Pearson IC、Rank IC；
- IC 均值、ICIR、t 值；
- 5 分位收益、最高组减最低组收益；
- 分位组合换手率；
- 每个持有期的逐日明细和汇总 CSV。

`next_open` 的 `forward_5d` 定义为：信号日下一交易日开盘买入，再过 5 个全市场交易日的开盘卖出。个股缺行不会把“5 日”误算成该股票的后 5 条记录。

当持有期跨过多个相邻信号（例如日频信号配 20 日收益）时，默认用自动阶数的 Newey–West/HAC 修正 ICIR 和 t 值；汇总中同时保留带 `_naive` 后缀的独立样本口径。可用 `--set analytics.hac_lags=0` 关闭修正，或填非负整数指定阶数。

### 批量筛选因子

先用一段历史作为训练期，结果只是一张筛选表，不会自动回测：

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

切换为 Lasso 只改方法和惩罚强度：

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

结果在 `<output>/factor_selection/selection.csv`。筛选期与后续回测期应分开，避免用未来收益挑因子。

### 两层筛选：固定划分与滚动验证

固定划分按 `[split]` 执行“单因子 IC → Lasso → 验证期选参数”，不会读取
`test/live` 区间：

```bash
uv run ashare-lab validate-factors \
  --config configs/csi1000.toml \
  --source 基本面因子 \
  --rebalance monthly \
  --split fixed \
  --output outputs/fixed_validation
```

滚动验证使用相同的样本外边界，只在 train+validation 开发区间内部滚动：

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

时间既可以修改 TOML，也可以直接覆盖：

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

程序会根据 `selection.horizon` 和成交方式自动删除收益标签跨越区间边界的
末尾信号。参数候选由 `selection.ic_thresholds`、`selection.alpha_grid` 控制。
结果在 `<output>/factor_validation/`，其中 `split.csv` 和 `folds.csv` 用于审计
时间划分，`candidates.csv` 用于比较参数，`factor_stability.csv` 检查方向稳定性，
`selected_factors.csv` 是最终入选结果。

## 4. 三种交易规则档位

| 档位 | 停牌/涨跌停 | T+1 | 费用 | 适合场景 |
|---|---|---|---|---|
| `minimal` | 不拦截；但缺价永远不能成交 | 关闭 | 强制为 0 | 快速看信号上限、排查研究逻辑 |
| `standard` | 默认停牌双向禁单；涨停禁买、跌停禁卖 | 默认开启 | 默认历史费用 | 常规 A 股研究，可用 `--set` 临时覆盖 |
| `custom` | 逐项配置，并可加插件 | 逐项配置 | 逐项配置/插件 | 特殊账户、特殊成交假设 |

`minimal` 会强制关闭停牌/涨跌停拦截、T+1 和全部费用，命令行里的相反设置也不会把它变复杂。`standard` 提供默认真实规则，但允许用 `--set` 覆盖某一项。`custom` 与 `standard` 的默认值相同，更适合提醒自己“本次规则是逐项显式设定的”：

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

费用选项支持：

- `historical`：内置历史费率；
- `none`：关闭该项；
- 数字：固定费率，例如 `0.0005`；
- `模块:函数`：完全自定义成本模型。

查看一条命令最终采用了什么配置：

```bash
uv run ashare-lab show-config \
  --config configs/csi1000.toml \
  --rebalance monthly \
  --execution same_close \
  --rules minimal
```

## 5. 还能在命令行改变什么

所有 TOML 字段都可以用 `--set 路径=值` 覆盖，而且可以重复写。

| 想得到的效果 | 参数 |
|---|---|
| 周频 / 月频 | `--rebalance weekly` / `--rebalance monthly` |
| 每 10 个交易日调仓 | `--rebalance every_n_days --set schedule.every_n_days=10` |
| 当日收盘 / 次日开盘 | `--execution same_close` / `--execution next_open` |
| 极简 / 标准 / 自定义规则 | `--rules minimal` / `standard` / `custom` |
| 改回测区间 | `--start 2020-01-01 --end 2025-12-31` |
| 改指数代码 | `--set research.benchmark=000852.SH` |
| 成分快照保守滞后 1 日 | `--set research.membership_lag=1` |
| 最高 N 只等权 | `--set portfolio.method=top_n_equal --set portfolio.top_n=100` |
| 最高 N 只按指数权重 | `--set portfolio.method=top_n_benchmark` |
| 指数权重倾斜 | `--set portfolio.method=benchmark_tilt --set portfolio.tilt=0.5` |
| 关闭行业中性 | `--set preprocess.neutralize_industry=false` |
| 关闭市值中性 | `--set preprocess.neutralize_size=false` |
| 缺因子时至少保留 2 个 | `--set preprocess.min_factor_count=2` |
| 调仓日全无有效分数时停止 / 持仓不动 / 清仓 | `--set portfolio.empty_signal=error` / `hold` / `cash` |
| 修改分位数 | `--set analytics.quantiles=10` |
| 修改输出目录 | `--output outputs/my_run` |
| Polars / DuckDB 数据后端 | `--data-engine polars` / `--data-engine duckdb` |
| DuckDB 每批因子数 | `--set data.batch_size=16` |
| DuckDB 查询线程数 | `--set data.duckdb_threads=4` |
| 整手下单示例 | `--set execution.lot_rounder=ashare_lab.execution:round_to_board_lot` |

完整默认配置见 [`configs/csi1000.toml`](configs/csi1000.toml)。长期使用的方案建议复制一份 TOML；临时实验再用命令行覆盖。

`portfolio.empty_signal` 默认是 `error`，避免数据缺失被悄悄解释为继续持仓；确认这是你的研究假设后，再改为 `hold` 或 `cash`。

`round_to_board_lot` 只是便于复制的 100 股向下取整示例，买卖都会取整；公司行动产生零股后“一次性卖出零股”等细则请改写 `execution.lot_rounder` 插件。默认不取整，适合先做纯权重研究。

当前引擎每天只接受一批完整目标权重，并按股票净差“先卖后买”，天然不会产生同一股票当日买后又卖；所以周/月频下 `T+1` 是面向未来日内/多批订单扩展保留的显式开关，现有单批结果通常不会因它改变。

## 6. 输出文件

回测输出到 `<output>/backtest/`：

| 文件 | 内容 |
|---|---|
| `summary.json` | 策略、基准、主动收益和本次运行摘要 |
| `nav.csv` | 对齐后的策略净值、基准净值和超额净值 |
| `nav.svg` | 策略、基准和超额净值走势图 |
| `drawdown.svg` | 策略、基准和超额净值回撤图 |
| `strategy_daily.csv` | 每日净值、现金、持仓市值、单边换手、双边成交额换手、费用 |
| `benchmark_daily.csv` | 基准组合的每日净值、现金、持仓市值、换手、费用 |
| `trades.parquet` | 每笔成交和费用 |
| `rejections.parquet` | 因停牌、涨跌停、缺价、现金等未成交的订单 |
| `positions.parquet` | 默认只在调仓日记录的实际持仓 |
| `targets.parquet` | 策略希望达到的目标权重 |
| `signal_panel.parquet` | 调仓日因子、标准化值和合成分数 |
| `resolved_config.json` | 本次完整配置，便于复现 |

单因子结果输出到 `<output>/single_factor/`。

因子筛选输出到 `<output>/factor_selection/`：

| 文件 | 内容 |
|---|---|
| `selection.csv` | 每个因子的覆盖率、IC 或 Lasso 系数以及是否入选 |
| `selected_factors.txt` | 入选因子简称，一行一个 |
| `resolved_config.json` | 本次实际使用的完整配置 |

## 7. 代码分层：想到新方法时改哪里

| 想改变的环节 | 文件 | 最小入口 |
|---|---|---|
| 原始字段名、读取方式、指数成分 as-of | `src/ashare_lab/data.py` | `DataCatalog` / `FactorCatalog` |
| DuckDB 扫描、过滤、连接和分批读取 | `src/ashare_lab/duckdb_data.py` | `DuckDBDataCatalog` / `DuckDBFactorCatalog` |
| 因子滞后、缩尾、中性化、标准化 | `src/ashare_lab/signals.py` | `preprocess_factors` |
| IC、Lasso 或自定义因子筛选 | `src/ashare_lab/selection.py` | `select_factors` |
| 净值和回撤图、自定义绘图 | `src/ashare_lab/charts.py` | `write_backtest_charts` |
| 多因子合成 | `src/ashare_lab/signals.py` | `combine_scores` |
| 周/月/自定义调仓日 | `src/ashare_lab/schedule.py` | `make_schedule` |
| Top N、指数权重倾斜 | `src/ashare_lab/portfolio.py` | `build_targets` |
| 稳定的撮合公共入口 | `src/ashare_lab/execution.py` | `run_execution` |
| 佣金、印花税、过户费及自定义成本 | `src/ashare_lab/costs.py` | `HistoricalAshareCosts` / `CostContext` |
| 撮合配置、输入校验、信号日映射 | `src/ashare_lab/execution_inputs.py` | `ExecutionConfig` / `prepare_market` |
| 先卖后买、交易限制、现金与每日记账 | `src/ashare_lab/execution_engine.py` | `run_engine` |
| IC、分组收益 | `src/ashare_lab/analytics.py` | `analyze_factor` |
| 净值指标 | `src/ashare_lab/metrics.py` | `performance_metrics` |
| 整条任务怎么串起来 | `src/ashare_lab/pipeline.py` | `run_factor_research` / `run_backtest` |
| 命令行参数 | `src/ashare_lab/cli.py` | `build_parser` |

主流程可以概括为：

```text
按列读取数据
  → point-in-time 中证1000成分
  → 信息可得滞后
  → 每个因子独立预处理
  → 多因子合成
  → 目标权重
  → 开盘/收盘撮合与规则
  → 净值、成交、拒单和报告
```

## 8. 插件：不改框架也能加方法

配置项写 `模块:函数` 即可载入。可安装包内的 [`src/ashare_lab/example_plugins.py`](src/ashare_lab/example_plugins.py) 放了几个能直接运行、复制后修改的最小例子：

```bash
# 组合：每期选择分数最高的 20%，然后等权
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --set portfolio.method=ashare_lab.example_plugins:top_twenty_percent

# 因子合成：对当前可用的标准化因子等权平均
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --set combine.method=ashare_lab.example_plugins:average_available_factors

# 调仓日：每月第一个交易日
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --set schedule.method=ashare_lab.example_plugins:first_trading_day_each_month

# 可交易规则：拒绝 trade_status=N 的新股订单
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rules custom \
  --set rules.method=ashare_lab.example_plugins:block_new_stocks

# 整套成本模型：所有费用为零
uv run ashare-lab backtest \
  --config configs/csi1000.toml \
  --rules custom \
  --set costs.method=ashare_lab.example_plugins:zero_cost
```

每个插件只需遵守输入输出约定：

| 环节 | 输入 | 输出 |
|---|---|---|
| 调仓 | `calendar, config` | `list[date]`，日期必须来自输入交易日历 |
| 整套预处理 | `panel, factor_specs, config` | 含 `date, stock, factor__<alias>` 的 Polars `DataFrame` |
| 因子合成 | `panel, factor_specs, config` | 带 `score` 的 `DataFrame`，或 `Series` / `Expr` |
| 组合 | `panel, config` | 含 `date, stock, target_weight` 的 `DataFrame`，每期非负且权重和为 1 |
| 可交易规则 | 一个 `OrderContext` | `None/True` 表示放行，`False` 或字符串表示拒单 |
| 整套成本 | `date, side, value`；或一个 `CostContext` | 一个 `TradeCost`；也可用带 `calculate(...)` 的对象 |
| 整手 | `requested_shares, side, stock, date` | 实际股数 |

对应配置键为 `schedule.method`、`preprocess.method`、`combine.method`、`portfolio.method`、`rules.method`、`costs.method`、`execution.lot_rounder`。

`costs.method` 替换的是整套成本模型，函数返回 `TradeCost`；`costs.stamp_tax` 和 `costs.transfer_fee` 只替换一个费用分项，函数返回费用金额 `float`。不要把两种签名混用。

需要按股票代码区分交易所、早期过户费或冲击成本时，让整套成本函数只接收一个 `CostContext`，即可读取 `context.stock/date/side/value`。内置过户费只覆盖统一口径和 2022-04-29 的费率边界；研究更早历史时应使用这个接口补充交易所差异。

## 9. 当前数据适配与防未来函数

框架已经按本项目真实 Parquet 结构处理，而不是按两个 txt 文件写死：

- 行情是 2004-12-31 至 2026-05-27 的股票长表；交易日历只取行情表；
- 估值和行业是自然日数据，不能拿它们生成交易日历；
- 中证1000 `000852.SH` 是 2014-10-31 起的月末快照，不是每日成分；每个信号日只向后查找 `snapshot_date <= signal_date` 的最新快照；
- 因子按 2012–2026 年和类别分文件，读取时只开请求年份和请求列；
- 多源因子以可投资池左连接，不会因深度学习/高频因子缺键而偷偷缩小股票池；
- `availability_lag` 在完整日频行情键上先移动，再筛周频/月频，`lag=1` 不会误变成“滞后一周/月”；
- 收益和持仓使用复权价，实际成交金额及整手接口使用原价；
- `same_close` 会明确提示潜在前视；`next_open` 严格映射到下一全市场交易日；
- `next_open` 默认不拿成交日收盘后才完整可知的涨跌停状态拦截开盘单；若想采用更保守的近似，可设 `--set rules.use_eod_limit_for_open=true`；
- 回测截止日后才会成交的末尾信号自动丢弃，不会让净值越过用户填写的结束日。

回测里的基准是用历史成分权重和复权股价重建的无摩擦代理组合，不是交易所发布的官方指数净值；`excess_nav` 因而表示相对这个可审计代理组合的净值比。

仍需谨慎：当前数据没有公告时间、ST 历史、完整证券主表和精确盘中涨跌停轨迹。基本面/一致预期因子的严格 point-in-time 性不能仅从现有文件证明；严谨研究应设置合适的 `availability_lag`，并补充公告时点数据。日线 `up_down_limit_status` 适用于收盘成交，或用户主动开启的开盘保守近似；更精确的盘中可交易性可用 `rules.method` 自定义。

长时间停牌、退市和恢复交易也缺少完整证券主表：当前持仓会沿用最后可得收盘价，极端退出案例应结合 `rejections.parquet` 单独审计并补自定义规则。

## 10. A 股规则来源与框架口径

标准模式内置了可替换的历史印花税表：2005-01-24、2007-05-30、2008-04-24、2008-09-19、2023-08-28 为变更边界。相关一手资料：

- [财政部：2005-01-24 调整为双边 1‰](https://www.mof.gov.cn/gkml/caizhengwengao/caizhengbuwengao2005/caizhengbuwengao20053/200805/t20080525_42735.htm)
- [国家税务总局：2007-05-30 调整为双边 3‰](https://www.chinatax.gov.cn/chinatax/n810341/n810765/n812176/200705/c1194505/content.html)
- [财政部：2008-04-24 调整为双边 1‰](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200805/t20080519_29133.htm)
- [财政部：2008-09-19 改为卖方单边 1‰](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/200809/t20080919_76432.htm)
- [税务总局：2023-08-28 起减半征收](https://zhejiang.chinatax.gov.cn/art/2023/8/28/art_8409_83672.html)
- [上交所交易规则：买入证券交收前不得卖出](https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml)

交易所规则会变化，所以框架优先使用数据中的停牌/涨跌停状态，并把规则、费用和整手全部留作配置或插件，而不是把板块阈值散落在主循环里。

## 11. 测试

```bash
uv run pytest

# 只跑不依赖真实大数据的 toy 端到端测试
uv run pytest tests/test_end_to_end.py
```

测试覆盖：

- toy 数据从 Parquet 到净值的端到端周频/月频测试；
- 指数月度快照严格 backward as-of；
- 因子列裁剪、Inf 清理和多源左连接；
- 因子 IC、分位收益和次日开盘标签对齐；
- 当日收盘/次日开盘、周末跳转；
- 停牌、涨跌停、特殊交易状态、缺价；
- T+1、现金守恒、最低佣金、全部印花税边界；
- `minimal / standard / custom` 的实际行为差异；
- 自定义费用和可交易规则插件。

## 12. 全A估计、中证1000应用的多因子组合

建议在 VS Code 中打开项目文件夹，然后点击“终端 → 新建终端”。默认开发配置只跑到
验证集末日，不会自动偷看样本外测试集。

```bash
# 最大化 IC_IR：风险矩阵来自最近 12 个已完成持有期的全A RankIC
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic_ir

# 最大化 IC：风险矩阵改用当期全A因子暴露相关矩阵
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic

# 改成周频，其他口径不变
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic_ir \
  --rebalance weekly
```

两种方法都满足因子权重非负且和为 1。默认先在每个因子类别内部优化，再把各类别的
标准化得分等权平均，避免量价因子仅因数量更多而支配总分。每个信号日只允许使用
`exit_date <= signal_date` 的历史 IC；尚未结束的持有期绝不会进入权重。

当前配置使用上一阶段冻结的 32 因子清单
`configs/csi1000_selected_32.csv`，在全A截面完成去极值、市值和行业中性化以及标准化，
但只在当期中证1000历史成分内生成股票分数。分数转成股票权重仍由
`portfolio.method` 决定，因此可以独立比较以下选股方式：

```bash
# 温和指数增强：指数权重按综合分倾斜（默认）
uv run ashare-lab multi-factor --config configs/csi1000_multifactor.toml \
  --set portfolio.method=benchmark_tilt

# 综合分最高的100只等权
uv run ashare-lab multi-factor --config configs/csi1000_multifactor.toml \
  --set portfolio.method=top_n_equal --set portfolio.top_n=100

# 最高100只按其原指数权重重新归一化
uv run ashare-lab multi-factor --config configs/csi1000_multifactor.toml \
  --set portfolio.method=top_n_benchmark --set portfolio.top_n=100
```

项目根目录还提供了“综合分前20%等权”的独立脚本。它保留因子合成、交易和费用
口径，只替换股票权重插件，默认依次运行最大化 IC_IR 和最大化 IC：

```bash
uv run python run_top20_equal.py

# 只运行一种因子合成方法
uv run python run_top20_equal.py --combination max_ic
```

本配置的成交口径是次日开盘，只拦停牌，买入和卖出各扣 0.1%，不再叠加印花税和
过户费。临时修改仍用 `--execution`、`--rules` 或 `--set costs.xxx=...`，无需改代码。

当前项目把2020年以后统一作为连续历史模拟实盘，可用以下命令跑到实际最新日期：

```bash
uv run ashare-lab multi-factor \
  --config configs/csi1000_multifactor.toml \
  --combination max_ic_ir \
  --end 2026-05-27 \
  --backtest-start 2020-11-01 \
  --output outputs/csi1000_sim_live_2020_latest
```

这是历史回放意义上的模拟实盘，不是真正在当时逐日落盘的实盘记录。如果根据这段
历史表现反复修改策略，应如实称为历史回测优化。

每次运行会在输出目录的 `multifactor/<方法>/` 下生成净值图、回撤图、逐年收益、
因子 IC 历史、逐期因子权重、优化诊断、目标权重、成交、持仓、拒单和完整配置。

未来接入非线性回归或机器学习时，把 `multifactor.method` 写为 `模块:函数`。函数签名为：

```python
def model(estimation_panel, application_panel, factor_specs, config):
    ...
    return scores  # Polars DataFrame，至少含 date、stock、score
```

`estimation_panel` 是带 `forward_return` 和 `exit_date` 的全A样本，
`application_panel` 是中证1000成分截面。自定义模型也必须在每个信号日过滤
`exit_date <= signal_date` 后再训练。股票权重、成交、费用和报告模块不需要随模型改动。
