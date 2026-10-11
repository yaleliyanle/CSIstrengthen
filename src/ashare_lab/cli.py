"""CLI entry point: common settings have dedicated options; override others with ``--set``."""

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
        description="A-share single-factor and multifactor research and index-enhancement backtesting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("show-config", "Show the resolved configuration"),
        ("data-info", "Inspect data files and factor sources without backtesting"),
        ("list-factors", "List available factor names"),
        ("single-factor", "Run single-factor IC and quantile diagnostics"),
        ("select-factors", "Select factors using IC or Lasso"),
        ("validate-factors", "Validate IC-to-Lasso selection with fixed splits or rolling folds"),
        ("multi-factor", "Run a dynamic multifactor strategy estimated on all A-shares and applied to the target index"),
        ("backtest", "Run a multifactor CSI 1000 enhancement backtest"),
    ]:
        command = subparsers.add_parser(name, help=help_text)
        _common_arguments(command)
        if name == "list-factors":
            command.add_argument("--source", help="Inspect one factor source directory (use its original dataset name)")
            command.add_argument("--contains", help="Show factors whose names contain this text")
        if name == "select-factors":
            command.add_argument(
                "--source",
                action="append",
                help="Use all factors in a source directory; repeat to include multiple sources",
            )
            command.add_argument("--contains", help="Keep factors whose names contain this text")
            command.add_argument("--method", choices=["ic", "lasso"])
            command.add_argument("--horizon", type=int, help="Forward-return horizon in trading days")
        if name == "validate-factors":
            command.add_argument(
                "--source",
                action="append",
                help="Use all factors in a source directory; repeat to include multiple sources",
            )
            command.add_argument("--contains", help="Keep factors whose names contain this text")
            command.add_argument("--split", choices=["fixed", "rolling"])
            for option, label in [
                ("train-start", "Training start date"),
                ("train-end", "Training end date"),
                ("validation-start", "Validation start date"),
                ("validation-end", "Validation end date"),
                ("test-start", "Out-of-sample start date"),
                ("test-end", "Out-of-sample end date"),
                ("live-start", "Simulated-live start date"),
                ("live-end", "Simulated-live end date"),
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
                help="Multifactor combination method",
            )
            command.add_argument("--factor-file", help="Frozen factor-list CSV")
            command.add_argument("--lookback", type=int, help="Number of periods in the historical IC window")
            command.add_argument(
                "--backtest-start", help="First signal date for strategy NAV, YYYY-MM-DD"
            )
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="TOML configuration file")
    parser.add_argument("--start", help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", help="End date, YYYY-MM-DD")
    parser.add_argument(
        "--rebalance",
        choices=["daily", "weekly", "monthly", "quarterly", "every_n_days"],
        help="Rebalance frequency",
    )
    parser.add_argument(
        "--execution", choices=["same_close", "next_open"], help="Execution timing"
    )
    parser.add_argument(
        "--rules",
        choices=["minimal", "standard", "custom"],
        help="Trading profile: minimal, standard A-share, or individually customized",
    )
    parser.add_argument("--output", help="Output root directory")
    parser.add_argument(
        "--data-engine",
        choices=["polars", "duckdb"],
        help="Parquet data backend; defaults to data.engine in TOML",
    )
    parser.add_argument(
        "--factor",
        action="append",
        metavar="SOURCE:NAME[:DIRECTION[:WEIGHT]]",
        help="Replace configured factors; repeat to add factors; quote complex names",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any setting; repeat as needed, e.g. --set portfolio.tilt=0.5",
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
            print(f"Completed {len(reports)} single-factor diagnostics.")
            print(f"Output directory: {Path(config.output['directory']) / 'single_factor'}")
        elif args.command == "select-factors":
            result = run_factor_selection(config)
            print(f"Selection method: {result.method}")
            print(f"Selected {len(result.selected)} factors: {', '.join(result.selected) or 'none'}")
            print(f"Output directory: {Path(config.output['directory']) / 'factor_selection'}")
        elif args.command == "validate-factors":
            result = run_factor_validation(config)
            print(f"Validation method: {result.method}")
            print(f"Best IC threshold: {result.best_ic_threshold}")
            print(f"Best Lasso alpha: {result.best_alpha}")
            print(f"Final selection of {len(result.selected)} factors: {', '.join(result.selected) or 'none'}")
            print(f"Output directory: {Path(config.output['directory']) / 'factor_validation'}")
        elif args.command == "multi-factor":
            result = run_multifactor_backtest(config)
            print(f"Combination method: {result.method}")
            print(json.dumps(result.summary, ensure_ascii=False, indent=2, default=str))
            print(f"Output directory: {result.output_directory}")
        elif args.command == "backtest":
            artifacts = run_backtest(config)
            print(json.dumps(artifacts.summary, ensure_ascii=False, indent=2, default=str))
            print(f"Output directory: {artifacts.output_directory}")
        return 0
    except Exception as error:
        if args.debug:
            raise
        parser.exit(2, f"Error: {error}\nUse --debug to see the full traceback.\n")


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
            raise ValueError("--source and --factor cannot be used together")
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
            raise ValueError("--source/--contains did not match any factors")
    return Config.from_dict(apply_overrides(raw, list(args.set)))


def _parse_factor(text: str) -> dict[str, Any]:
    parts = text.split(":")
    if len(parts) not in {2, 3, 4} or not parts[0] or not parts[1]:
        raise ValueError(
            f"--factor must use SOURCE:NAME[:DIRECTION[:WEIGHT]]; received {text!r}"
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
    print("Market and auxiliary data")
    print(files)
    print("\nFactor data")
    print(factors.info().with_columns(
        (pl.col("size_bytes") / 1024**3).round(3).alias("size_gb")
    ).drop("size_bytes"))


def _print_factors(config: Config, source: str | None, contains: str | None) -> None:
    _, catalog = make_catalogs(config)
    result = catalog.list_factors(source)
    if contains:
        result = result.filter(pl.col("name").str.contains(contains, literal=True))
    print(result)
    print(f"Total: {result.height} factors")


def _effective_rules(config: Config) -> str:
    profile = config.rules.get("profile", "standard")
    if profile == "minimal":
        return "\nEffective rules: minimal (requires a price only; no suspension/price-limit/T+1 restrictions; zero costs)"
    enabled = [
        name
        for name in ("block_suspended", "block_limit", "t_plus_one")
        if config.rules.get(name, True)
    ]
    open_limit = "; apply end-of-day price-limit status at the open too" if config.rules.get(
        "use_eod_limit_for_open", False
    ) else ""
    return (
        f"\nEffective rules: {profile} (enabled: {', '.join(enabled) or 'none'}"
        f"; costs from [costs]{open_limit})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
