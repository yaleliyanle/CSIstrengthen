"""DuckDB data backend.

This module handles Parquet scanning, row/column filtering, and joins. The research layer uses
the same ``date`` / ``stock`` interface as Polars, preserving strategy semantics across backends.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import duckdb
import polars as pl

from .data import KEYS, MARKET_FIELDS, VALUATION_FIELDS, DateLike


def _day(value: DateLike) -> date:
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


def _quote(name: str) -> str:
    if not name or "\x00" in name:
        raise ValueError(f"Invalid field name: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _frame(connection: duckdb.DuckDBPyConnection, sql: str, params=()) -> pl.DataFrame:
    """Execute SQL and return a Polars DataFrame directly."""

    return connection.execute(sql, params).pl()


def _schema(connection: duckdb.DuckDBPyConnection, path: Path) -> dict[str, str]:
    rows = connection.execute(
        "describe select * from read_parquet(?)", [str(path)]
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


@dataclass
class DuckDBDataCatalog:
    """DuckDB implementation of the :class:`DataCatalog` interface."""

    root: str | Path = Path("a share")
    # Match the same original dataset filenames as the Polars catalog.
    market_name: str = "A股行情数据.parquet"
    valuation_name: str = "A股估值数据.parquet"
    index_name: str = "指数成分股.parquet"
    industry_name: str = "A股行业分类.parquet"
    threads: int = 4
    _con: duckdb.DuckDBPyConnection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._con = duckdb.connect()
        self._con.execute(f"pragma threads={int(self.threads)}")

    def _path(self, name: str) -> Path:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def _panel(
        self,
        path: Path,
        aliases: Mapping[str, str],
        columns: Sequence[str] | None,
        start: DateLike | None,
        end: DateLike | None,
        stocks: Sequence[str] | None,
    ) -> pl.DataFrame:
        raw_types = _schema(self._con, path)
        for key in ("trade_dt", "s_info_windcode"):
            if key not in raw_types:
                raise ValueError(f"{path.name} is missing key field {key!r}")
        canonical = {raw: aliases.get(raw, raw) for raw in raw_types}
        reverse = {name: raw for raw, name in canonical.items()}
        available = [canonical[name] for name in raw_types]
        wanted = (
            [name for name in available if name not in KEYS]
            if columns is None
            else [name for name in _unique(columns) if name not in KEYS]
        )
        unknown = [name for name in wanted if name not in reverse]
        if unknown:
            raise KeyError(f"Unknown fields {unknown}; available fields: {available}")

        projection = [
            "cast(trade_dt as date) as date",
            "cast(s_info_windcode as varchar) as stock",
            *[
                f"{_quote(reverse[name])} as {_quote(name)}"
                for name in wanted
            ],
        ]
        clauses: list[str] = []
        params: list[object] = [str(path)]
        if start is not None:
            clauses.append("cast(trade_dt as date) >= ?")
            params.append(_day(start))
        if end is not None:
            clauses.append("cast(trade_dt as date) <= ?")
            params.append(_day(end))
        if stocks is not None:
            stock_frame = pl.DataFrame({"stock": _unique(stocks)})
            self._con.register("__ashare_stocks", stock_frame)
            clauses.append(
                "cast(s_info_windcode as varchar) in "
                "(select stock from __ashare_stocks)"
            )
        where = f" where {' and '.join(clauses)}" if clauses else ""
        try:
            return _frame(
                self._con,
                f"select {', '.join(projection)} from read_parquet(?) {where}",
                params,
            )
        finally:
            if stocks is not None:
                self._con.unregister("__ashare_stocks")

    def market(self, **kwargs: object) -> pl.DataFrame:
        return self._panel(
            self._path(self.market_name),
            MARKET_FIELDS,
            kwargs.get("columns"),
            kwargs.get("start"),
            kwargs.get("end"),
            kwargs.get("stocks"),
        )

    load_market = market
    prices = market

    def scan_market(self, **kwargs: object) -> pl.LazyFrame:
        return self.market(**kwargs).lazy()

    def valuation(self, **kwargs: object) -> pl.DataFrame:
        return self._panel(
            self._path(self.valuation_name),
            VALUATION_FIELDS,
            kwargs.get("columns"),
            kwargs.get("start"),
            kwargs.get("end"),
            kwargs.get("stocks"),
        )

    load_valuation = valuation

    def scan_valuation(self, **kwargs: object) -> pl.LazyFrame:
        return self.valuation(**kwargs).lazy()

    def index_members(
        self,
        dates: Sequence[DateLike],
        index: str = "000852.SH",
        strict: bool = True,
    ) -> pl.DataFrame:
        requested = sorted({_day(value) for value in dates})
        if not requested:
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "index": pl.String,
                    "stock": pl.String,
                    "index_weight": pl.Float64,
                    "snapshot_date": pl.Date,
                }
            )
        self._con.register("__ashare_dates", pl.DataFrame({"date": requested}))
        path = str(self._path(self.index_name))
        try:
            result = _frame(
                self._con,
                """
                with snapshots as (
                    select distinct cast(trade_dt as date) as snapshot_date
                    from read_parquet(?)
                    where cast(s_info_windcode as varchar) = ?
                ), mapped as (
                    select d.date, max(s.snapshot_date) as snapshot_date
                    from __ashare_dates d
                    left join snapshots s on s.snapshot_date <= d.date
                    group by d.date
                )
                select
                    m.date,
                    cast(i.s_info_windcode as varchar) as index,
                    cast(i.s_con_windcode as varchar) as stock,
                    cast(i.i_weight as double) as index_weight,
                    m.snapshot_date
                from mapped m
                join read_parquet(?) i
                  on cast(i.trade_dt as date) = m.snapshot_date
                 and cast(i.s_info_windcode as varchar) = ?
                order by m.date, stock
                """,
                [path, index, path, index],
            )
        finally:
            self._con.unregister("__ashare_dates")
        missing = sorted(set(requested) - set(result["date"].to_list()))
        if strict and missing:
            raise ValueError(f"No earlier index snapshots for these dates: {missing}")
        return result

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
        path = self._path(self.industry_name)
        available = _schema(self._con, path)
        wanted = _unique(stocks)
        missing = [stock for stock in wanted if stock not in available]
        if strict and missing:
            raise KeyError(f"The industry table is missing stock columns: {missing}")
        wanted = [stock for stock in wanted if stock in available]
        if not wanted:
            return pl.DataFrame(
                schema={"date": pl.Date, "stock": pl.String, "industry": pl.String}
            )

        clauses: list[str] = []
        params: list[object] = [str(path)]
        if start is not None:
            clauses.append("cast(trade_dt as date) >= ?")
            params.append(_day(start))
        if end is not None:
            clauses.append("cast(trade_dt as date) <= ?")
            params.append(_day(end))
        if dates is not None:
            self._con.register(
                "__ashare_industry_dates",
                pl.DataFrame({"date": [_day(value) for value in dates]}),
            )
            clauses.append(
                "cast(trade_dt as date) in "
                "(select date from __ashare_industry_dates)"
            )
        where = f" where {' and '.join(clauses)}" if clauses else ""
        projection = ", ".join(_quote(stock) for stock in wanted)
        try:
            wide = _frame(
                self._con,
                f"select cast(trade_dt as date) as date, {projection} "
                f"from read_parquet(?) {where}",
                params,
            )
        finally:
            if dates is not None:
                self._con.unregister("__ashare_industry_dates")
        result = wide.unpivot(
            on=wanted,
            index="date",
            variable_name="stock",
            value_name="industry",
        )
        if drop_nulls:
            result = result.filter(pl.col("industry").is_not_null())
        return result.sort(KEYS)

    load_industry = industry

    def info(self) -> pl.DataFrame:
        rows = []
        for dataset, name in (
            ("market", self.market_name),
            ("valuation", self.valuation_name),
            ("index", self.index_name),
            ("industry", self.industry_name),
        ):
            path = self.root / name
            if path.exists():
                rows.append(
                    {
                        "dataset": dataset,
                        "path": str(path),
                        "columns": len(_schema(self._con, path)),
                        "size_bytes": path.stat().st_size,
                    }
                )
        return pl.DataFrame(rows)


@dataclass
class DuckDBFactorCatalog:
    """Scan annual Parquet files in factor batches and join the stock universe inside DuckDB."""

    root: str | Path = Path("factor")
    batch_size: int = 16
    threads: int = 4
    _files: dict[str, dict[int, Path]] = field(init=False, default_factory=dict)
    _schemas: dict[str, dict[str, str]] = field(init=False, default_factory=dict)
    _con: duckdb.DuckDBPyConnection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._con = duckdb.connect()
        self._con.execute(f"pragma threads={int(self.threads)}")
        year_at_end = re.compile(r"(\d{4})(?=\.parquet$)")
        for path in sorted(self.root.glob("*/*.parquet")):
            match = year_at_end.search(path.name)
            if not match:
                continue
            source, year = path.parent.name, int(match.group(1))
            files = self._files.setdefault(source, {})
            if year in files:
                raise ValueError(f"{source} has multiple factor files for {year}")
            files[year] = path
        for source, files in self._files.items():
            schema = _schema(self._con, files[min(files)])
            missing = {"trade_dt", "s_info_windcode"} - set(schema)
            if missing:
                raise ValueError(f"{source} is missing key fields: {sorted(missing)}")
            self._schemas[source] = schema

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._files))

    def _check_source(self, source: str) -> None:
        if source not in self._files:
            raise KeyError(f"Unknown factor source {source!r}; available: {list(self.sources)}")

    def factor_names(self, source: str) -> tuple[str, ...]:
        self._check_source(source)
        return tuple(
            name
            for name in self._schemas[source]
            if name not in {"trade_dt", "s_info_windcode"}
        )

    def list_factors(self, source: str | None = None) -> pl.DataFrame:
        sources = [source] if source else list(self.sources)
        rows = []
        for item in sources:
            self._check_source(item)
            for name in self.factor_names(item):
                rows.append(
                    {"source": item, "name": name, "dtype": self._schemas[item][name]}
                )
        return pl.DataFrame(
            rows,
            schema={"source": pl.String, "name": pl.String, "dtype": pl.String},
        )

    def info(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "source": source,
                    "first_year": min(files),
                    "last_year": max(files),
                    "files": len(files),
                    "factors": len(self.factor_names(source)),
                    "size_bytes": sum(path.stat().st_size for path in files.values()),
                }
                for source, files in self._files.items()
            ]
        )

    def _paths(
        self,
        source: str,
        start: DateLike | None,
        end: DateLike | None,
        years: Sequence[int] | None = None,
    ) -> list[str]:
        self._check_source(source)
        files = self._files[source]
        if years is not None:
            wanted = sorted(set(years))
            missing = [year for year in wanted if year not in files]
            if missing:
                raise FileNotFoundError(f"{source} is missing annual files: {missing}")
        else:
            first = _day(start).year if start is not None else min(files)
            last = _day(end).year if end is not None else max(files)
            wanted = [year for year in sorted(files) if first <= year <= last]
        paths = [str(files[year]) for year in wanted]
        if not paths:
            raise FileNotFoundError(f"{source} has no annual files within the requested dates")
        return paths

    @staticmethod
    def _factor_expr(name: str, prefix: str = "f") -> str:
        value = f"try_cast({prefix}.{_quote(name)} as double)"
        return (
            f"case when isfinite({value}) then {value} else null end "
            f"as {_quote(name)}"
        )

    def load(
        self,
        source: str,
        names: str | Sequence[str],
        start: DateLike | None = None,
        end: DateLike | None = None,
        stocks: Sequence[str] | None = None,
        years: Sequence[int] | None = None,
    ) -> pl.DataFrame:
        names = [names] if isinstance(names, str) else _unique(names)
        unknown = [name for name in names if name not in self.factor_names(source)]
        if unknown:
            raise KeyError(f"{source} has no factors {unknown}")
        paths = self._paths(source, start, end, years)
        clauses: list[str] = []
        params: list[object] = [paths]
        if start is not None:
            clauses.append("cast(f.trade_dt as date) >= ?")
            params.append(_day(start))
        if end is not None:
            clauses.append("cast(f.trade_dt as date) <= ?")
            params.append(_day(end))
        if stocks is not None:
            self._con.register("__factor_stocks", pl.DataFrame({"stock": _unique(stocks)}))
            clauses.append(
                "cast(f.s_info_windcode as varchar) in "
                "(select stock from __factor_stocks)"
            )
        where = f" where {' and '.join(clauses)}" if clauses else ""
        projection = ", ".join(self._factor_expr(name) for name in names)
        try:
            return _frame(
                self._con,
                "select cast(f.trade_dt as date) as date, "
                "cast(f.s_info_windcode as varchar) as stock"
                + (f", {projection}" if projection else "")
                + f" from read_parquet(?, union_by_name=true) f {where}",
                params,
            )
        finally:
            if stocks is not None:
                self._con.unregister("__factor_stocks")

    def scan(self, *args: object, **kwargs: object) -> pl.LazyFrame:
        return self.load(*args, **kwargs).lazy()

    def load_many(
        self,
        requests: Mapping[str, str | Sequence[str]],
        start: DateLike | None = None,
        end: DateLike | None = None,
        stocks: Sequence[str] | None = None,
        universe: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> pl.DataFrame:
        """Left-join batches inside DuckDB, preserving the original Polars interface."""

        normalized = self._normalize_requests(requests)
        if universe is None:
            keys = [
                self.load(source, names[:1], start, end, stocks).select(KEYS)
                for source, names in normalized.items()
            ]
            base = pl.concat(keys).unique().sort(KEYS)
        else:
            base = universe.collect() if isinstance(universe, pl.LazyFrame) else universe
            missing = set(KEYS) - set(base.columns)
            if missing:
                raise ValueError(f"universe is missing key fields: {sorted(missing)}")
            if start is not None:
                base = base.filter(pl.col("date") >= _day(start))
            if end is not None:
                base = base.filter(pl.col("date") <= _day(end))
            if stocks is not None:
                base = base.filter(pl.col("stock").is_in(_unique(stocks)))
        return self._join_batches(base, normalized, start, end)

    load_factors = load_many

    def load_many_for_signals(
        self,
        requests: Mapping[str, str | Sequence[str]],
        lags: Mapping[str, int],
        signal_dates: Sequence[DateLike],
        universe: pl.DataFrame,
        start: DateLike,
        end: DateLike,
    ) -> pl.DataFrame:
        """Compute lags on the full trading-day grid and return only rebalance rows to the research layer."""

        normalized = self._normalize_requests(requests)
        dates = sorted({_day(value) for value in signal_dates})
        grid = universe.select(KEYS).filter(
            (pl.col("date") >= _day(start)) & (pl.col("date") <= _day(end))
        ).unique().sort(KEYS)
        anchor = grid.filter(pl.col("date").is_in(dates))
        self._con.register("__factor_grid", grid)
        self._con.register("__factor_anchor", anchor)
        self._con.register("__factor_signal_dates", pl.DataFrame({"date": dates}))
        result = anchor
        try:
            for source, names in normalized.items():
                for offset in range(0, len(names), self.batch_size):
                    batch = names[offset : offset + self.batch_size]
                    result = self._signal_batch(
                        result, source, batch, lags, start, end
                    )
        finally:
            self._con.unregister("__factor_signal_dates")
            self._con.unregister("__factor_anchor")
            self._con.unregister("__factor_grid")
        return result.sort(KEYS)

    def _normalize_requests(
        self, requests: Mapping[str, str | Sequence[str]]
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        seen: set[str] = set()
        for source, values in requests.items():
            names = [values] if isinstance(values, str) else _unique(values)
            unknown = [name for name in names if name not in self.factor_names(source)]
            if unknown:
                raise KeyError(f"{source} has no factors {unknown}")
            overlap = seen & set(names)
            if overlap:
                raise ValueError(f"Multi-source requests contain duplicate factor names: {sorted(overlap)}")
            seen.update(names)
            normalized[source] = names
        if not normalized:
            raise ValueError("requests must not be empty")
        return normalized

    def _join_batches(
        self,
        base: pl.DataFrame,
        requests: Mapping[str, list[str]],
        start: DateLike | None,
        end: DateLike | None,
    ) -> pl.DataFrame:
        work = base.with_row_index("__row_id")
        self._con.register("__factor_universe", work)
        try:
            result = work
            for source, names in requests.items():
                paths = self._paths(source, start, end)
                for offset in range(0, len(names), self.batch_size):
                    batch = names[offset : offset + self.batch_size]
                    projection = ", ".join(
                        self._factor_expr(name) for name in batch
                    )
                    joined = _frame(
                        self._con,
                        "with factors as (select cast(f.trade_dt as date) as date, "
                        "cast(f.s_info_windcode as varchar) as stock, "
                        f"{projection} from read_parquet(?, union_by_name=true) f) "
                        "select u.__row_id, "
                        + ", ".join(f"f.{_quote(name)}" for name in batch)
                        + " from __factor_universe u left join factors f "
                        "on f.date=u.date and f.stock=u.stock order by u.__row_id",
                        [paths],
                    )
                    if joined.height != work.height:
                        raise ValueError(f"{source} contains duplicate date/stock records")
                    result = result.with_columns(
                        [joined.get_column(name) for name in batch]
                    )
            return result.drop("__row_id")
        finally:
            self._con.unregister("__factor_universe")

    def _signal_batch(
        self,
        current: pl.DataFrame,
        source: str,
        names: list[str],
        lags: Mapping[str, int],
        start: DateLike,
        end: DateLike,
    ) -> pl.DataFrame:
        paths = self._paths(source, start, end)
        raw_projection = ", ".join(self._factor_expr(name) for name in names)
        lagged = any(int(lags.get(name, 0)) > 0 for name in names)
        if lagged:
            shifted = ", ".join(
                f"lag({_quote(name)}, {int(lags.get(name, 0))}) over "
                f"(partition by stock order by date) as {_quote(name)}"
                for name in names
            )
            sql = (
                "with factors as (select cast(f.trade_dt as date) as date, "
                "cast(f.s_info_windcode as varchar) as stock, "
                f"{raw_projection} from read_parquet(?, union_by_name=true) f "
                "where cast(f.trade_dt as date) between ? and ? "
                "and cast(f.s_info_windcode as varchar) in "
                "(select distinct stock from __factor_grid)), "
                "daily as (select g.date, g.stock, "
                + ", ".join(f"f.{_quote(name)}" for name in names)
                + " from __factor_grid g left join factors f "
                "on f.date=g.date and f.stock=g.stock), "
                f"shifted as (select date, stock, {shifted} from daily) "
                "select a.date, a.stock, "
                + ", ".join(f"s.{_quote(name)}" for name in names)
                + " from __factor_anchor a left join shifted s "
                "on s.date=a.date and s.stock=a.stock order by a.date, a.stock"
            )
            params = [paths, _day(start), _day(end)]
        else:
            sql = (
                "with factors as (select cast(f.trade_dt as date) as date, "
                "cast(f.s_info_windcode as varchar) as stock, "
                f"{raw_projection} from read_parquet(?, union_by_name=true) f "
                "where cast(f.trade_dt as date) in "
                "(select date from __factor_signal_dates) "
                "and cast(f.s_info_windcode as varchar) in "
                "(select distinct stock from __factor_anchor)) "
                "select a.date, a.stock, "
                + ", ".join(f"f.{_quote(name)}" for name in names)
                + " from __factor_anchor a left join factors f "
                "on f.date=a.date and f.stock=a.stock order by a.date, a.stock"
            )
            params = [paths]
        batch = _frame(self._con, sql, params)
        if batch.height != current.height:
            raise ValueError(f"{source} contains duplicate date/stock records")
        return current.join(batch, on=KEYS, how="left")


__all__ = ["DuckDBDataCatalog", "DuckDBFactorCatalog"]
