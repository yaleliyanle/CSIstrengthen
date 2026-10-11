"""CSI 1000 top-20% equal-weight strategy based on combined scores.

Use max_ic_ir / max_ic factor weights and replace only the final stock-weight layer.
Run from the VS Code terminal: ``uv run python run_top20_equal.py``.
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
    parser = argparse.ArgumentParser(description="Equal-weight backtest for the top-scoring 20% of stocks")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--combination",
        choices=["both", "max_ic_ir", "max_ic"],
        default="both",
        help="Factor combination method; defaults to running both",
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
            f"\n{method}: total excess return "
            f"{active['total_excess_return']:.2%}, "
            f"information ratio {active['information_ratio']:.3f}"
        )
        print(f"Output directory: {result.output_directory}")

    comparison = yearly_results[0]
    for yearly in yearly_results[1:]:
        comparison = comparison.join(yearly, on="year", how="full", coalesce=True)
    print("\nAnnual excess returns:")
    print(
        comparison.sort("year").with_columns(
            pl.exclude("year").mul(100).round(2)
        )
    )
    print("Values in percent (%)")


if __name__ == "__main__":
    main()
