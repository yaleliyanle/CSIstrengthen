"""Backtest charts using standard SVG without a heavyweight plotting dependency."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from .plugins import import_callable


def write_backtest_charts(
    nav: pl.DataFrame,
    directory: str | Path,
    cfg: Mapping[str, Any] | Any,
) -> list[Path]:
    """Generate NAV and drawdown charts; disable or replace the method in ``[charts]``."""

    options = _options(cfg)
    if not bool(options.get("enabled", True)):
        return []
    custom = options.get("method")
    if custom:
        result = import_callable(str(custom))(nav, Path(directory), cfg)
        return [Path(item) for item in result]

    series = _series(nav)
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    width = int(options.get("width", 1100))
    height = int(options.get("height", 500))
    paths = [folder / "nav.svg", folder / "drawdown.svg"]
    dates = nav.get_column("date").to_list()
    values = {
        label: nav.get_column(column).cast(pl.Float64).to_numpy()
        for label, column in series.items()
    }
    _write_svg(paths[0], dates, values, "Backtest NAV", _number, width, height)
    drawdowns = {label: _drawdown(item) for label, item in values.items()}
    _write_svg(paths[1], dates, drawdowns, "Drawdown", _percent, width, height)
    return paths


def _series(nav: pl.DataFrame) -> dict[str, str]:
    candidates = {
        "Strategy": "strategy_nav",
        "Benchmark": "benchmark_nav",
        "Excess": "excess_nav",
        "NAV": "net_value",
    }
    found = {label: column for label, column in candidates.items() if column in nav.columns}
    if not found or "date" not in nav.columns:
        raise ValueError("Charts require date and NAV columns")
    return found


def _write_svg(
    path: Path,
    dates: list[Any],
    series: dict[str, np.ndarray],
    title: str,
    formatter: Callable[[float], str],
    width: int,
    height: int,
) -> None:
    if not dates:
        raise ValueError("No NAV data available for plotting")
    left, right, top, bottom = 72, 72, 48, 80
    plot_width = max(width - left - right, 1)
    plot_height = max(height - top - bottom, 1)
    finite = np.concatenate([values[np.isfinite(values)] for values in series.values()])
    if not len(finite):
        raise ValueError("NAV data contains no finite values")
    low, high = float(finite.min()), float(finite.max())
    padding = max((high - low) * 0.06, 0.01)
    low, high = low - padding, high + padding
    start, end = dates[0].toordinal(), dates[-1].toordinal()
    span = max(end - start, 1)

    def x(day: Any) -> float:
        return left + (day.toordinal() - start) / span * plot_width

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    colors = ["#2563eb", "#f59e0b", "#10b981", "#7c3aed"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="18" font-weight="600">{escape(title)}</text>',
    ]
    for index in range(6):
        value = low + (high - low) * index / 5
        py = y(value)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_width}" y2="{py:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#4b5563">{escape(formatter(value))}</text>'
        )
    tick_count = min(6, len(dates))
    for index in np.linspace(0, len(dates) - 1, tick_count, dtype=int):
        px = x(dates[index])
        parts.append(
            f'<text x="{px:.1f}" y="{height - 43}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#4b5563">{dates[index]}</text>'
        )
    for color, (label, values) in zip(colors, series.items()):
        points = " ".join(
            f"{x(day):.1f},{y(float(value)):.1f}"
            for day, value in zip(dates, values)
            if np.isfinite(value)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    legend_x = left
    for color, label in zip(colors, series):
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 16}" x2="{legend_x + 22}" y2="{height - 16}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 28}" y="{height - 12}" font-family="sans-serif" font-size="12">{escape(label)}</text>',
            ]
        )
        legend_x += 105
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _drawdown(values: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(values)
    return np.divide(values, peaks, out=np.ones_like(values), where=peaks != 0) - 1


def _number(value: float) -> str:
    return f"{value:.2f}"


def _percent(value: float) -> str:
    return f"{value:.0%}"


def _options(cfg: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if hasattr(cfg, "charts"):
        return cfg.charts
    if isinstance(cfg, Mapping):
        value = cfg.get("charts")
        return value if isinstance(value, Mapping) else cfg
    raise TypeError("cfg must be Config or dict")


__all__ = ["write_backtest_charts"]
