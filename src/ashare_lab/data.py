"""Fast, narrow readers for the local A-share parquet files.

The public boundary always uses ``date`` and ``stock``.  Raw Wind field names
stay inside this module, which keeps the research code independent of storage.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import polars as pl


DateLike = str | date | datetime
KEYS = ["date", "stock"]


MARKET_FIELDS = {
    "trade_dt": "date",
    "s_info_windcode": "stock",
    "s_dq_open": "open",
    "s_dq_high": "high",
    "s_dq_low": "low",
    "s_dq_close": "close",
    "s_dq_avgprice": "vwap",
    "s_dq_pctchange": "pct_change",
    "s_dq_volume": "volume",
    "s_dq_adjvolume": "adj_volume",
    "s_dq_amount": "amount",
    "s_dq_adjamount": "adj_amount",
    "s_dq_adjpreclose": "adj_preclose",
    "s_dq_adjopen": "adj_open",
    "s_dq_adjhigh": "adj_high",
    "s_dq_adjlow": "adj_low",
    "s_dq_adjclose": "adj_close",
    "s_dq_adjfactor": "adj_factor",
    "s_dq_adjvwap": "adj_vwap",
    "s_dq_tradestatus": "trade_status",
}

VALUATION_FIELDS = {
    "trade_dt": "date",
    "s_info_windcode": "stock",
    "s_val_mv": "market_cap",
    "s_dq_mv": "float_market_cap",
    "s_val_pb_new": "pb",
    "s_val_pe_ttm": "pe_ttm",
    "s_val_pcf_ocfttm": "pcf_ocf_ttm",
    "s_val_pcf_ncfttm": "pcf_ncf_ttm",
    "s_val_ps_ttm": "ps_ttm",
    "s_dq_turn": "turnover",
    "s_dq_freeturnover": "free_turnover",
    "tot_shr_today": "total_shares",
    "float_a_shr_today": "float_a_shares",
    "s_price_div_dps": "price_div_dps",
    "free_shares_today": "free_shares",
    "net_profit_parent_comp_ttm": "net_profit_ttm",
    "up_down_limit_status": "limit_status",
}


def _as_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    return date.fromisoformat(text)


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _filter_dates(
    frame: pl.LazyFrame,
    column: str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    dates: Sequence[DateLike] | None = None,
) -> pl.LazyFrame:
    if start is not None:
        frame = frame.filter(pl.col(column) >= _as_date(start))
    if end is not None:
        frame = frame.filter(pl.col(column) <= _as_date(end))
    if dates is not None:
        frame = frame.filter(pl.col(column).is_in([_as_date(x) for x in dates]))
    return frame


def _filter_canonical(
    frame: pl.LazyFrame,
    start: DateLike | None,
    end: DateLike | None,
    stocks: Sequence[str] | None,
) -> pl.LazyFrame:
    frame = _filter_dates(frame, "date", start, end)
    if stocks is not None:
        frame = frame.filter(pl.col("stock").is_in(_unique(stocks)))
    return frame


def _empty_index_members() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "date": pl.Date,
            "index": pl.String,
            "stock": pl.String,
            "index_weight": pl.Float64,
            "snapshot_date": pl.Date,
        }
    )


@dataclass
class DataCatalog:
    """Access the four datasets in ``a share`` with canonical field names."""

    root: str | Path = Path("a share")
    market_name: str = "A股行情数据.parquet"
    valuation_name: str = "A股估值数据.parquet"
    index_name: str = "指数成分股.parquet"
    industry_name: str = "A股行业分类.parquet"

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def _path(self, name: str) -> Path:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _scan_panel(
        self,
        path: Path,
        aliases: Mapping[str, str],
        columns: Sequence[str] | None,
        start: DateLike | None,
        end: DateLike | None,
        stocks: Sequence[str] | None,
    ) -> pl.LazyFrame:
        frame = pl.scan_parquet(path)
        raw_names = frame.collect_schema().names()
        for key in ("trade_dt", "s_info_windcode"):
            if key not in raw_names:
                raise ValueError(f"{path.name} 缺少键字段 {key!r}")

        canonical = {raw: aliases.get(raw, raw) for raw in raw_names}
        reverse = {name: raw for raw, name in canonical.items()}
        available = [canonical[x] for x in raw_names]
        wanted = (
            [x for x in available if x not in KEYS]
            if columns is None
            else [x for x in _unique(columns) if x not in KEYS]
        )
        unknown = [x for x in wanted if x not in reverse]
        if unknown:
            raise KeyError(f"未知字段 {unknown}; 可用字段: {available}")

        raw_values = [reverse[x] for x in wanted]
        frame = frame.select(["trade_dt", "s_info_windcode", *raw_values])
        frame = _filter_dates(frame, "trade_dt", start, end)
        if stocks is not None:
            frame = frame.filter(
                pl.col("s_info_windcode").is_in(_unique(stocks))
            )
        return frame.rename({x: canonical[x] for x in frame.collect_schema().names()})

    def scan_market(
        self,
        columns: Sequence[str] | None = None,
        start: DateLike | None = None,
        end: DateLike | None = None,
        stocks: Sequence[str] | None = None,
    ) -> pl.LazyFrame:
        """Lazily read only the requested market rows and columns."""

        return self._scan_panel(
            self._path(self.market_name),
            MARKET_FIELDS,
            columns,
            start,
            end,
            stocks,
        )

    def market(self, **kwargs: object) -> pl.DataFrame:
        return self.scan_market(**kwargs).collect()

    load_market = market
    prices = market

    def scan_valuation(
        self,
        columns: Sequence[str] | None = None,
        start: DateLike | None = None,
        end: DateLike | None = None,
        stocks: Sequence[str] | None = None,
    ) -> pl.LazyFrame:
        """Lazily read only the requested valuation rows and columns."""

        return self._scan_panel(
            self._path(self.valuation_name),
            VALUATION_FIELDS,
            columns,
            start,
            end,
            stocks,
        )

    def valuation(self, **kwargs: object) -> pl.DataFrame:
        return self.scan_valuation(**kwargs).collect()

    load_valuation = valuation

    def index_members(
        self,
        dates: Sequence[DateLike],
        index: str = "000852.SH",
        strict: bool = True,
    ) -> pl.DataFrame:
        """Expand monthly index snapshots using the latest snapshot on/before each date.

        ``snapshot_date`` is retained so callers and tests can audit that no future
        constituents leaked into a requested date.
        """

        requested_dates = sorted({_as_date(x) for x in dates})
        if not requested_dates:
            return _empty_index_members()

        members = (
            pl.scan_parquet(self._path(self.index_name))
            .select(
                ["trade_dt", "s_info_windcode", "s_con_windcode", "i_weight"]
            )
            .filter(pl.col("s_info_windcode") == index)
            .filter(pl.col("trade_dt") <= requested_dates[-1])
            .rename(
                {
                    "trade_dt": "snapshot_date",
                    "s_info_windcode": "index",
                    "s_con_windcode": "stock",
                    "i_weight": "index_weight",
                }
            )
            .collect()
        )
        if members.is_empty():
            raise ValueError(
                f"{index} 在 {requested_dates[-1]} 前没有更早的成分股快照"
            )

        snapshots = members.select("snapshot_date").unique().sort("snapshot_date")
        mapping = pl.DataFrame({"date": requested_dates}).join_asof(
            snapshots,
            left_on="date",
            right_on="snapshot_date",
            strategy="backward",
        )
        missing = mapping.filter(pl.col("snapshot_date").is_null())["date"].to_list()
        if strict and missing:
            raise ValueError(f"这些日期没有更早的指数快照: {missing}")

        return (
            mapping.drop_nulls("snapshot_date")
            .join(members, on="snapshot_date", how="left")
            .select(["date", "index", "stock", "index_weight", "snapshot_date"])
            .sort(KEYS)
        )

    load_index_members = index_members

    def industry(
        self,
        stocks: Sequence[str],
        start: DateLike | None = None,
        end: DateLike | None = None,
        dates: Sequence[DateLike] | None = None,
        drop_nulls: bool = False,
        strict: bool = True,
    ) -> pl.DataFrame:
        """Project requested stock columns before converting the wide table to long."""

        stocks = _unique(stocks)
        path = self._path(self.industry_name)
        schema_names = pl.scan_parquet(path).collect_schema().names()
        missing = [stock for stock in stocks if stock not in schema_names]
        if strict and missing:
            raise KeyError(f"行业表没有这些股票列: {missing}")
        stocks = [stock for stock in stocks if stock in schema_names]
        if not stocks:
            return pl.DataFrame(
                schema={"date": pl.Date, "stock": pl.String, "industry": pl.String}
            )

        wide = pl.scan_parquet(path).select(["trade_dt", *stocks])
        wide = _filter_dates(wide, "trade_dt", start, end, dates).collect()
        long = wide.rename({"trade_dt": "date"}).unpivot(
            on=stocks,
            index="date",
            variable_name="stock",
            value_name="industry",
        )
        if drop_nulls:
            long = long.filter(pl.col("industry").is_not_null())
        return long.sort(KEYS)

    load_industry = industry

    def info(self) -> pl.DataFrame:
        """Return schema-level metadata without scanning table contents."""

        rows = []
        specs = [
            ("market", self.market_name, MARKET_FIELDS),
            ("valuation", self.valuation_name, VALUATION_FIELDS),
            ("index", self.index_name, {}),
            ("industry", self.industry_name, {}),
        ]
        for dataset, name, aliases in specs:
            path = self.root / name
            if not path.exists():
                continue
            raw = pl.scan_parquet(path).collect_schema().names()
            rows.append(
                {
                    "dataset": dataset,
                    "path": str(path),
                    "columns": len(raw),
                    "canonical_columns": [aliases.get(x, x) for x in raw],
                    "size_bytes": path.stat().st_size,
                }
            )
        return pl.DataFrame(rows)


@dataclass
class FactorCatalog:
    """Discover and selectively read yearly factor parquet files."""

    root: str | Path = Path("factor")
    _files: dict[str, dict[int, Path]] = field(init=False, default_factory=dict)
    _schemas: dict[str, pl.Schema] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        year_at_end = re.compile(r"(\d{4})(?=\.parquet$)")
        for path in sorted(self.root.glob("*/*.parquet")):
            match = year_at_end.search(path.name)
            if not match:
                continue
            source, year = path.parent.name, int(match.group(1))
            files = self._files.setdefault(source, {})
            if year in files:
                raise ValueError(f"{source} 的 {year} 年有多个因子文件")
            files[year] = path

        for source, files in self._files.items():
            schema = pl.scan_parquet(files[min(files)]).collect_schema()
            missing = {"trade_dt", "s_info_windcode"} - set(schema.names())
            if missing:
                raise ValueError(f"{source} 缺少键字段: {sorted(missing)}")
            self._schemas[source] = schema

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._files))

    def _check_source(self, source: str) -> None:
        if source not in self._files:
            raise KeyError(f"未知因子源 {source!r}; 可用: {list(self.sources)}")

    def factor_names(self, source: str) -> tuple[str, ...]:
        self._check_source(source)
        return tuple(
            x
            for x in self._schemas[source].names()
            if x not in {"trade_dt", "s_info_windcode"}
        )

    def list_factors(self, source: str | None = None) -> pl.DataFrame:
        """List factor names from parquet schemas only; no data pages are read."""

        sources = [source] if source is not None else list(self.sources)
        rows = []
        for item in sources:
            self._check_source(item)
            schema = self._schemas[item]
            for name in self.factor_names(item):
                rows.append(
                    {"source": item, "name": name, "dtype": str(schema[name])}
                )
        return pl.DataFrame(
            rows,
            schema={"source": pl.String, "name": pl.String, "dtype": pl.String},
        )

    def info(self) -> pl.DataFrame:
        """Summarise sources using filenames, schemas and filesystem metadata."""

        rows = []
        for source in self.sources:
            files = self._files[source]
            rows.append(
                {
                    "source": source,
                    "first_year": min(files),
                    "last_year": max(files),
                    "files": len(files),
                    "factors": len(self.factor_names(source)),
                    "size_bytes": sum(path.stat().st_size for path in files.values()),
                }
            )
        return pl.DataFrame(rows)

    def _paths(
        self,
        source: str,
        start: DateLike | None,
        end: DateLike | None,
        years: Sequence[int] | None,
    ) -> list[Path]:
        self._check_source(source)
        files = self._files[source]
        if years is not None:
            wanted = sorted(set(years))
            missing = [year for year in wanted if year not in files]
            if missing:
                raise FileNotFoundError(f"{source} 缺少年度文件: {missing}")
        else:
            first = _as_date(start).year if start is not None else min(files)
            last = _as_date(end).year if end is not None else max(files)
            wanted = [year for year in sorted(files) if first <= year <= last]
        paths = [files[year] for year in wanted]
        if not paths:
            raise FileNotFoundError(f"{source} 在请求日期内没有年度文件")
        return paths

    def scan(
        self,
        source: str,
        names: str | Sequence[str],
        start: DateLike | None = None,
        end: DateLike | None = None,
        stocks: Sequence[str] | None = None,
        years: Sequence[int] | None = None,
    ) -> pl.LazyFrame:
        """Read only two keys and requested factors from relevant yearly files."""

        names = [names] if isinstance(names, str) else _unique(names)
        available = set(self.factor_names(source))
        unknown = [name for name in names if name not in available]
        if unknown:
            raise KeyError(f"{source} 没有因子 {unknown}")

        paths = self._paths(source, start, end, years)
        frame = pl.scan_parquet([str(path) for path in paths]).select(
            ["trade_dt", "s_info_windcode", *names]
        )
        frame = _filter_dates(frame, "trade_dt", start, end)
        if stocks is not None:
            frame = frame.filter(
                pl.col("s_info_windcode").is_in(_unique(stocks))
            )
        frame = frame.rename(
            {"trade_dt": "date", "s_info_windcode": "stock"}
        )

        schema = self._schemas[source]
        floats = [
            name for name in names if schema[name] in {pl.Float32, pl.Float64}
        ]
        if floats:
            frame = frame.with_columns(
                [
                    pl.when(pl.col(name).is_finite())
                    .then(pl.col(name))
                    .otherwise(None)
                    .alias(name)
                    for name in floats
                ]
            )
        return frame

    def load(self, *args: object, **kwargs: object) -> pl.DataFrame:
        return self.scan(*args, **kwargs).collect()

    def load_many(
        self,
        requests: Mapping[str, str | Sequence[str]],
        start: DateLike | None = None,
        end: DateLike | None = None,
        stocks: Sequence[str] | None = None,
        universe: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        """Left-join several sources onto a universe (or the union of their keys)."""

        scans: list[pl.LazyFrame] = []
        requested_names: list[str] = []
        for source, names in requests.items():
            names = [names] if isinstance(names, str) else _unique(names)
            overlap = set(requested_names) & set(names)
            if overlap:
                raise ValueError(f"多源请求含重名因子: {sorted(overlap)}")
            requested_names.extend(names)
            scans.append(self.scan(source, names, start, end, stocks))

        if not scans and universe is None:
            raise ValueError("requests 不能为空，除非提供 universe")

        if universe is not None:
            base = universe.lazy() if isinstance(universe, pl.DataFrame) else universe
            base_names = base.collect_schema().names()
            missing = set(KEYS) - set(base_names)
            if missing:
                raise ValueError(f"universe 缺少键字段: {sorted(missing)}")
            collision = set(base_names) & set(requested_names)
            if collision:
                raise ValueError(f"universe 已有同名因子: {sorted(collision)}")
            base = _filter_canonical(base, start, end, stocks)
        else:
            base = pl.concat([scan.select(KEYS) for scan in scans]).unique()

        for scan in scans:
            base = base.join(scan, on=KEYS, how="left")
        return base.sort(KEYS).collect()

    load_factors = load_many
