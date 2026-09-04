"""批量比较不同指数中的 TOP20% 等权策略。"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ashare_lab.multifactor_pipeline import run_multifactor_backtest
from ashare_lab.settings import load_config


INDICES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000906.SH": "中证800",
}
METHODS = ("max_ic_ir", "max_ic")
PORTFOLIO = "ashare_lab.example_plugins:top_twenty_percent"


def main() -> None:
    parser = argparse.ArgumentParser(description="跨指数 TOP20% 等权实验")
    parser.add_argument("--config", default="configs/csi1000_multifactor.toml")
    parser.add_argument("--backtest-start", default="2020-11-01")
    parser.add_argument("--end", default="2026-05-27")
    parser.add_argument("--output", default="outputs/cross_index_top20")
    parser.add_argument(
        "--index",
        action="append",
        choices=list(INDICES),
        help="只跑指定指数，可重复；默认三个都跑",
    )
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    yearly_frames: list[pl.DataFrame] = []
    selected = args.index or list(INDICES)

    for index_code in selected:
        index_name = INDICES[index_code]
        index_output = Path(args.output) / index_code.replace(".", "_")
        for method in METHODS:
            config = load_config(
                args.config,
                [
                    f"research.benchmark={index_code}",
                    f"research.end={args.end}",
                    f"multifactor.backtest_start={args.backtest_start}",
                    f"multifactor.method={method}",
                    f"portfolio.method={PORTFOLIO}",
                    f"output.directory={index_output}",
                ],
            )
            result = run_multifactor_backtest(config)
            summary = result.summary
            observations = int(summary["strategy"]["observations"])
            total_excess = float(summary["active"]["total_excess_return"])
            annual_excess = (1.0 + total_excess) ** (252 / observations) - 1.0
            daily = result.strategy.daily_nav
            annual_turnover = float(daily["turnover"].sum()) / (observations / 252)
            positions = result.targets.group_by("date").len()["len"]
            summary_rows.append(
                {
                    "index_code": index_code,
                    "index_name": index_name,
                    "method": method,
                    "start": summary["run"]["start"],
                    "end": summary["run"]["end"],
                    "target_stocks": int(positions.median()),
                    "strategy_return": summary["strategy"]["total_return"],
                    "benchmark_return": summary["benchmark"]["total_return"],
                    "total_excess_return": total_excess,
                    "annual_excess_return": annual_excess,
                    "information_ratio": summary["active"]["information_ratio"],
                    "tracking_error": summary["active"]["tracking_error"],
                    "max_drawdown": summary["strategy"]["max_drawdown"],
                    "annual_half_turnover": annual_turnover,
                    "fees": summary["run"]["fees"],
                }
            )
            yearly_frames.append(
                pl.read_csv(
                    result.output_directory / "yearly_performance.csv"
                ).with_columns(
                    pl.lit(index_code).alias("index_code"),
                    pl.lit(index_name).alias("index_name"),
                    pl.lit(method).alias("method"),
                ).select(
                    "index_code",
                    "index_name",
                    "method",
                    "year",
                    "strategy_return",
                    "benchmark_return",
                    "excess_return",
                )
            )
            print(
                f"{index_name} {method}: 累计超额 {total_excess:.2%}，"
                f"年化超额 {annual_excess:.2%}"
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary_table = pl.DataFrame(summary_rows).sort("index_code", "method")
    yearly_table = pl.concat(yearly_frames).sort("index_code", "method", "year")
    summary_table.write_csv(output / "cross_index_summary.csv")
    yearly_table.write_csv(output / "cross_index_yearly.csv")
    print("\n汇总：")
    print(
        summary_table.select(
            "index_name",
            "method",
            pl.col("total_excess_return").mul(100).round(2).alias("累计超额%"),
            pl.col("annual_excess_return").mul(100).round(2).alias("年化超额%"),
            pl.col("information_ratio").round(3),
        )
    )
    print(f"结果目录：{output}")


if __name__ == "__main__":
    main()
