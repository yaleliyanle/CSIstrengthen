"""命令行入口；常用参数有短选项，其余一律用 ``--set`` 覆盖。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from .pipeline import (
    make_catalogs,
    run_backtest,
    run_factor_research,
    run_factor_selection,
    run_factor_validation,
)
from .multifactor_pipeline import run_multifactor_backtest
from .settings import Config, apply_overrides, load_config


DEFAULT_CONFIG = "configs/csi1000.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ashare-lab",
        description="A 股单因子、多因子与指数增强回测",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("show-config", "显示最终生效的配置"),
        ("data-info", "查看数据文件和因子源，不跑回测"),
        ("list-factors", "列出可用因子名称"),
        ("single-factor", "运行单因子 IC 和分组检验"),
        ("select-factors", "用 IC 或 Lasso 筛选一组因子"),
        ("validate-factors", "固定划分或滚动验证 IC→Lasso 两层筛选"),
        ("multi-factor", "运行全A估计、目标指数应用的动态多因子策略"),
        ("backtest", "运行多因子中证1000增强回测"),
    ]:
        command = subparsers.add_parser(name, help=help_text)
        _common_arguments(command)
        if name == "list-factors":
            command.add_argument("--source", help="只看一个因子目录，例如 基本面因子")
            command.add_argument("--contains", help="只看名称包含该文字的因子")
        if name == "select-factors":
            command.add_argument(
                "--source",
                action="append",
                help="使用该目录中的全部因子，可重复，例如 基本面因子",
            )
            command.add_argument("--contains", help="只保留名称包含该文字的因子")
            command.add_argument("--method", choices=["ic", "lasso"])
            command.add_argument("--horizon", type=int, help="前瞻收益日数")
        if name == "validate-factors":
            command.add_argument(
                "--source",
                action="append",
                help="使用该目录中的全部因子，可重复，例如 基本面因子",
            )
            command.add_argument("--contains", help="只保留名称包含该文字的因子")
            command.add_argument("--split", choices=["fixed", "rolling"])
            for option, label in [
                ("train-start", "训练开始日"),
                ("train-end", "训练结束日"),
                ("validation-start", "验证开始日"),
                ("validation-end", "验证结束日"),
                ("test-start", "样本外开始日"),
                ("test-end", "样本外结束日"),
                ("live-start", "模拟实盘开始日"),
                ("live-end", "模拟实盘结束日"),
            ]:
                command.add_argument(f"--{option}", help=f"{label} YYYY-MM-DD")
            command.add_argument(
                "--rolling-window", choices=["expanding", "sliding"]
            )
            command.add_argument("--rolling-train-months", type=int)
            command.add_argument("--rolling-validation-months", type=int)
            command.add_argument("--rolling-step-months", type=int)
        if name == "multi-factor":
            command.add_argument(
                "--combination",
                choices=["equal", "max_ic_ir", "max_ic"],
                help="多因子合成方法",
            )
            command.add_argument("--factor-file", help="冻结因子清单CSV")
            command.add_argument("--lookback", type=int, help="历史IC窗口期数")
            command.add_argument(
                "--backtest-start", help="策略净值开始信号日 YYYY-MM-DD"
            )
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="TOML 配置文件")
    parser.add_argument("--start", help="开始日 YYYY-MM-DD")
    parser.add_argument("--end", help="结束日 YYYY-MM-DD")
    parser.add_argument(
        "--rebalance",
        choices=["daily", "weekly", "monthly", "quarterly", "every_n_days"],
        help="调仓频率",
    )
    parser.add_argument(
        "--execution", choices=["same_close", "next_open"], help="成交时点"
    )
    parser.add_argument(
        "--rules",
        choices=["minimal", "standard", "custom"],
        help="交易规则档位：极简、标准A股、逐项自定义",
    )
    parser.add_argument("--output", help="输出根目录")
    parser.add_argument(
        "--data-engine",
        choices=["polars", "duckdb"],
        help="Parquet 数据后端；默认读取 TOML 中的 data.engine",
    )
    parser.add_argument(
        "--factor",
        action="append",
        metavar="SOURCE:NAME[:DIRECTION[:WEIGHT]]",
        help="替换配置因子，可重复；复杂名称请加引号",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖任意配置，可重复，例如 --set portfolio.tilt=0.5",
    )
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.command == "show-config":
            print(config.to_json(effective=True))
            print(_effective_rules(config))
        elif args.command == "data-info":
            _print_data_info(config)
        elif args.command == "list-factors":
            _print_factors(config, args.source, args.contains)
        elif args.command == "single-factor":
            reports = run_factor_research(config)
            print(f"完成 {len(reports)} 组单因子检验。")
            print(f"结果目录: {Path(config.output['directory']) / 'single_factor'}")
        elif args.command == "select-factors":
            result = run_factor_selection(config)
            print(f"筛选方法：{result.method}")
            print(f"入选 {len(result.selected)} 个：{', '.join(result.selected) or '无'}")
            print(f"结果目录: {Path(config.output['directory']) / 'factor_selection'}")
        elif args.command == "validate-factors":
            result = run_factor_validation(config)
            print(f"验证方式：{result.method}")
            print(f"最佳 IC 门槛：{result.best_ic_threshold}")
            print(f"最佳 Lasso alpha：{result.best_alpha}")
            print(f"最终入选 {len(result.selected)} 个：{', '.join(result.selected) or '无'}")
            print(f"结果目录: {Path(config.output['directory']) / 'factor_validation'}")
        elif args.command == "multi-factor":
            result = run_multifactor_backtest(config)
            print(f"合成方法：{result.method}")
            print(json.dumps(result.summary, ensure_ascii=False, indent=2, default=str))
            print(f"结果目录: {result.output_directory}")
        elif args.command == "backtest":
            artifacts = run_backtest(config)
            print(json.dumps(artifacts.summary, ensure_ascii=False, indent=2, default=str))
            print(f"结果目录: {artifacts.output_directory}")
        return 0
    except Exception as error:
        if args.debug:
            raise
        parser.exit(2, f"错误: {error}\n使用 --debug 可查看完整堆栈。\n")


def _config_from_args(args: argparse.Namespace) -> Config:
    direct = {
        "research.start": args.start,
        "research.end": args.end,
        "schedule.method": args.rebalance,
        "execution.price": args.execution,
        "rules.profile": args.rules,
        "output.directory": args.output,
        "data.engine": args.data_engine,
        "selection.method": getattr(args, "method", None),
        "selection.horizon": getattr(args, "horizon", None),
        "multifactor.method": getattr(args, "combination", None),
        "multifactor.factor_file": getattr(args, "factor_file", None),
        "multifactor.lookback_periods": getattr(args, "lookback", None),
        "multifactor.backtest_start": getattr(args, "backtest_start", None),
        "split.method": getattr(args, "split", None),
        "split.train_start": getattr(args, "train_start", None),
        "split.train_end": getattr(args, "train_end", None),
        "split.validation_start": getattr(args, "validation_start", None),
        "split.validation_end": getattr(args, "validation_end", None),
        "split.test_start": getattr(args, "test_start", None),
        "split.test_end": getattr(args, "test_end", None),
        "split.live_start": getattr(args, "live_start", None),
        "split.live_end": getattr(args, "live_end", None),
        "rolling.window": getattr(args, "rolling_window", None),
        "rolling.train_months": getattr(args, "rolling_train_months", None),
        "rolling.validation_months": getattr(
            args, "rolling_validation_months", None
        ),
        "rolling.step_months": getattr(args, "rolling_step_months", None),
    }
    direct_overrides = [f"{key}={value}" for key, value in direct.items() if value]
    config = load_config(args.config, direct_overrides)
    raw = config.to_dict()
    if args.factor:
        raw["factors"] = [_parse_factor(value) for value in args.factor]
    sources = getattr(args, "source", None)
    contains = getattr(args, "contains", None)
    if args.command in {"select-factors", "validate-factors"} and sources:
        if args.factor:
            raise ValueError("--source 和 --factor 不能同时使用")
        _, catalog = make_catalogs(config)
        discovered = catalog.list_factors()
        discovered = discovered.filter(pl.col("source").is_in(sources))
        if contains:
            discovered = discovered.filter(
                pl.col("name").str.contains(contains, literal=True)
            )
        raw["factors"] = [
            {"source": row["source"], "name": row["name"]}
            for row in discovered.iter_rows(named=True)
        ]
        if not raw["factors"]:
            raise ValueError("--source/--contains 没有匹配到因子")
    return Config.from_dict(apply_overrides(raw, list(args.set)))


def _parse_factor(text: str) -> dict[str, Any]:
    parts = text.split(":")
    if len(parts) not in {2, 3, 4} or not parts[0] or not parts[1]:
        raise ValueError(
            f"--factor 应写成 SOURCE:NAME[:DIRECTION[:WEIGHT]]，收到 {text!r}"
        )
    result: dict[str, Any] = {"source": parts[0], "name": parts[1]}
    if len(parts) >= 3:
        result["direction"] = float(parts[2])
    if len(parts) == 4:
        result["weight"] = float(parts[3])
    return result


def _print_data_info(config: Config) -> None:
    data, factors = make_catalogs(config)
    files = data.info().select("dataset", "path", "columns", "size_bytes").with_columns(
        (pl.col("size_bytes") / 1024**3).round(3).alias("size_gb")
    ).drop("size_bytes")
    print("行情与辅助数据")
    print(files)
    print("\n因子数据")
    print(factors.info().with_columns(
        (pl.col("size_bytes") / 1024**3).round(3).alias("size_gb")
    ).drop("size_bytes"))


def _print_factors(config: Config, source: str | None, contains: str | None) -> None:
    _, catalog = make_catalogs(config)
    result = catalog.list_factors(source)
    if contains:
        result = result.filter(pl.col("name").str.contains(contains, literal=True))
    print(result)
    print(f"共 {result.height} 个因子")


def _effective_rules(config: Config) -> str:
    profile = config.rules.get("profile", "standard")
    if profile == "minimal":
        return "\n有效规则: minimal（只要求价格存在；无停牌/涨跌停/T+1限制；费用为0）"
    enabled = [
        name
        for name in ("block_suspended", "block_limit", "t_plus_one")
        if config.rules.get(name, True)
    ]
    open_limit = "；开盘也使用日终涨跌停字段" if config.rules.get(
        "use_eod_limit_for_open", False
    ) else ""
    return (
        f"\n有效规则: {profile}（开启 {', '.join(enabled) or '无'}"
        f"；费用按 [costs]{open_limit}）"
    )


if __name__ == "__main__":
    raise SystemExit(main())
