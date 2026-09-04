from datetime import date

import polars as pl

from ashare_lab.data import DataCatalog, FactorCatalog
from ashare_lab.duckdb_data import DuckDBDataCatalog, DuckDBFactorCatalog


def _market_files(root):
    root.mkdir()
    pl.DataFrame(
        {
            "trade_dt": [date(2024, 1, 2), date(2024, 1, 3)],
            "s_info_windcode": ["A.SZ", "B.SH"],
            "s_dq_close": [10.0, 20.0],
        }
    ).write_parquet(root / "A股行情数据.parquet")
    pl.DataFrame(
        {
            "trade_dt": [date(2024, 1, 2), date(2024, 1, 3)],
            "s_info_windcode": ["A.SZ", "B.SH"],
            "s_val_mv": [100.0, 200.0],
        }
    ).write_parquet(root / "A股估值数据.parquet")


def test_duckdb_and_polars_panel_readers_agree(tmp_path):
    root = tmp_path / "a share"
    _market_files(root)
    polars = DataCatalog(root)
    duckdb = DuckDBDataCatalog(root)

    expected = polars.market(
        columns=["close"], start="2024-01-02", stocks=["A.SZ"]
    )
    actual = duckdb.market(
        columns=["close"], start="2024-01-02", stocks=["A.SZ"]
    )
    assert actual.to_dicts() == expected.to_dicts()


def test_duckdb_batches_and_lags_on_the_full_trading_grid(tmp_path):
    factor_root = tmp_path / "factor"
    source = factor_root / "demo"
    source.mkdir(parents=True)
    pl.DataFrame(
        {
            "trade_dt": [
                date(2024, 1, 2),
                date(2024, 1, 4),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ],
            "s_info_windcode": ["A.SZ", "A.SZ", "B.SH", "B.SH", "B.SH"],
            "factor_a": [1.0, 3.0, 10.0, 20.0, 30.0],
            "factor_b": [2.0, 4.0, 11.0, 21.0, 31.0],
        }
    ).write_parquet(source / "daily_demo_2024.parquet")
    grid = pl.DataFrame(
        {
            "date": [date(2024, 1, day) for day in (2, 3, 4) for _ in range(2)],
            "stock": ["A.SZ", "B.SH"] * 3,
        }
    )

    duckdb = DuckDBFactorCatalog(factor_root, batch_size=1)
    direct = duckdb.load_many(
        {"demo": ["factor_a", "factor_b"]}, universe=grid
    )
    polars = FactorCatalog(factor_root).load_many(
        {"demo": ["factor_a", "factor_b"]}, universe=grid
    )
    assert direct.to_dicts() == polars.to_dicts()

    lagged = duckdb.load_many_for_signals(
        {"demo": ["factor_a", "factor_b"]},
        lags={"factor_a": 1, "factor_b": 0},
        signal_dates=[date(2024, 1, 4)],
        universe=grid,
        start=date(2024, 1, 2),
        end=date(2024, 1, 4),
    )
    rows = {row["stock"]: row for row in lagged.to_dicts()}
    assert rows["A.SZ"]["factor_a"] is None  # 1月3日无记录，但仍算一个交易日
    assert rows["B.SH"]["factor_a"] == 20.0
    assert rows["A.SZ"]["factor_b"] == 4.0
