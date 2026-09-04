"""中证1000综合分前20%等权策略。

因子权重仍由最大化 IC_IR / 最大化 IC 生成；本脚本只替换最后的股票权重层。
在 VS Code 终端直接运行：``uv run python run_top20_equal.py``。
"""

from __future__ import annotations

import argparse

import polars as pl

from ashare_lab.multifactor_pipeline import run_multifactor_backtest
from ashare_lab.settings import load_config


DEFAULT_CONFIG = "configs/csi1000_multifactor.toml"
DEFAULT_OUTPUT = "outputs/csi1000_top20_equal"
PORTFOLIO_PLUGIN = "ashare_lab.example_plugins:top_twenty_percent"


def main() -> None:
    parser = argparse.ArgumentParser(description="综合分前20%的股票等权回测")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--combination",
        choices=["both", "max_ic_ir", "max_ic"],
        default="both",
        help="因子合成方法；默认两种都跑",
    )
    parser.add_argument("--backtest-start", default="2020-11-01")
    parser.add_argument("--end", default="2026-05-27")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    methods = (
        ["max_ic_ir", "max_ic"]
        if args.combination == "both"
        else [args.combination]
    )
    yearly_results: list[pl.DataFrame] = []

    for method in methods:
        config = load_config(
            args.config,
            [
                f"research.end={args.end}",
                f"multifactor.backtest_start={args.backtest_start}",
                f"multifactor.method={method}",
                f"portfolio.method={PORTFOLIO_PLUGIN}",
                f"output.directory={args.output}",
            ],
        )
        result = run_multifactor_backtest(config)
        yearly_results.append(
            pl.read_csv(
                result.output_directory / "yearly_performance.csv"
            ).select(
                "year",
                pl.col("excess_return").alias(method),
            )
        )
        active = result.summary["active"]
        print(
            f"\n{method}: 累计超额 "
            f"{active['total_excess_return']:.2%}，"
            f"信息比率 {active['information_ratio']:.3f}"
        )
        print(f"结果目录：{result.output_directory}")

    comparison = yearly_results[0]
    for yearly in yearly_results[1:]:
        comparison = comparison.join(yearly, on="year", how="full", coalesce=True)
    print("\n逐年超额收益：")
    print(
        comparison.sort("year").with_columns(
            pl.exclude("year").mul(100).round(2)
        )
    )
    print("数值单位：%")


if __name__ == "__main__":
    main()
