from datetime import date

import polars as pl
import pytest

from ashare_lab.data import DataCatalog, FactorCatalog


def _write_data_files(root):
    root.mkdir()
    pl.DataFrame(
        {
            "s_info_windcode": ["A.SZ", "B.SH", "A.SZ"],
            "trade_dt": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3)],
            "s_dq_open": [10.0, 20.0, 11.0],
            "s_dq_close": [10.5, 19.0, 11.5],
            "unused": [1, 2, 3],
        }
    ).write_parquet(root / "A股行情数据.parquet")
    pl.DataFrame(
        {
            "s_info_windcode": ["A.SZ", "B.SH", "A.SZ"],
            "trade_dt": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3)],
            "s_val_mv": [100.0, 200.0, 110.0],
            "s_val_pb_new": [1.0, 2.0, 1.1],
        }
    ).write_parquet(root / "A股估值数据.parquet")
    pl.DataFrame(
        {
            "s_info_windcode": ["IDX", "IDX", "IDX", "IDX"],
            "s_con_windcode": ["A.SZ", "B.SH", "B.SH", "C.BJ"],
            "trade_dt": [
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 2, 29),
                date(2024, 2, 29),
            ],
            "i_weight": [0.6, 0.4, 0.5, 0.5],
            "__index_level_0__": [0, 1, 2, 3],
        }
    ).write_parquet(root / "指数成分股.parquet")
    pl.DataFrame(
        {
            "A.SZ": ["bank", "tech"],
            "B.SH": ["utility", "utility"],
            "C.BJ": [None, "health"],
            "trade_dt": [date(2024, 1, 2), date(2024, 1, 3)],
        }
    ).write_parquet(root / "A股行业分类.parquet")


def _write_factor(root, source, year, data):
    folder = root / source
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(folder / f"daily_{source}_{year}.parquet")


def test_market_and_valuation_project_rows_and_use_canonical_names(tmp_path):
    data_root = tmp_path / "a share"
    _write_data_files(data_root)
    catalog = DataCatalog(data_root)

    market = catalog.market(
        columns=["close"],
        start="2024-01-03",
        end="2024-01-03",
        stocks=["A.SZ"],
    )
    assert market.columns == ["date", "stock", "close"]
    assert market.to_dicts() == [
        {"date": date(2024, 1, 3), "stock": "A.SZ", "close": 11.5}
    ]

    valuation = catalog.valuation(columns=["market_cap"], stocks=["B.SH"])
    assert valuation.columns == ["date", "stock", "market_cap"]
    assert valuation["market_cap"].to_list() == [200.0]


def test_index_snapshots_expand_strictly_backward_without_future_members(tmp_path):
    data_root = tmp_path / "a share"
    _write_data_files(data_root)
    catalog = DataCatalog(data_root)

    result = catalog.index_members(
        [date(2024, 2, 15), date(2024, 3, 1)], index="IDX"
    )
    feb = result.filter(pl.col("date") == date(2024, 2, 15))
    march = result.filter(pl.col("date") == date(2024, 3, 1))

    assert set(feb["stock"]) == {"A.SZ", "B.SH"}
    assert feb["snapshot_date"].unique().to_list() == [date(2024, 1, 31)]
    assert "C.BJ" not in feb["stock"].to_list()
    assert set(march["stock"]) == {"B.SH", "C.BJ"}
    assert march["snapshot_date"].unique().to_list() == [date(2024, 2, 29)]

    with pytest.raises(ValueError, match="no constituent snapshot on or before"):
        catalog.index_members([date(2024, 1, 1)], index="IDX")


def test_industry_projects_stock_columns_before_unpivot(tmp_path):
    data_root = tmp_path / "a share"
    _write_data_files(data_root)
    catalog = DataCatalog(data_root)

    result = catalog.industry(
        stocks=["A.SZ", "C.BJ"], dates=[date(2024, 1, 3)]
    )

    assert result.columns == ["date", "stock", "industry"]
    assert result.to_dicts() == [
        {"date": date(2024, 1, 3), "stock": "A.SZ", "industry": "tech"},
        {"date": date(2024, 1, 3), "stock": "C.BJ", "industry": "health"},
    ]
    assert "B.SH" not in result["stock"].to_list()


def test_factor_projection_nonfinite_cleanup_and_cross_source_left_join(tmp_path):
    factor_root = tmp_path / "factor"
    common_dates = [date(2024, 1, 2), date(2024, 1, 2)]
    _write_factor(
        factor_root,
        "source_one",
        2024,
        {
            "trade_dt": common_dates,
            "s_info_windcode": ["A.SZ", "B.SH"],
            "factor_a": [1.0, float("inf")],
            "unused_a": [99.0, 99.0],
        },
    )
    _write_factor(
        factor_root,
        "source_two",
        2024,
        {
            "trade_dt": common_dates,
            "s_info_windcode": ["A.SZ", "C.BJ"],
            "factor_b": [10.0, 30.0],
            "unused_b": [88.0, 88.0],
        },
    )
    universe = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)] * 3,
            "stock": ["A.SZ", "B.SH", "C.BJ"],
            "index_weight": [0.4, 0.3, 0.3],
        }
    )

    catalog = FactorCatalog(factor_root)
    result = catalog.load_many(
        {"source_one": ["factor_a"], "source_two": ["factor_b"]},
        start="2024-01-02",
        end="2024-01-02",
        universe=universe,
    )

    assert catalog.sources == ("source_one", "source_two")
    assert set(catalog.list_factors()["name"]) == {
        "factor_a",
        "factor_b",
        "unused_a",
        "unused_b",
    }
    assert result.columns == [
        "date",
        "stock",
        "index_weight",
        "factor_a",
        "factor_b",
    ]
    rows = {row["stock"]: row for row in result.to_dicts()}
    assert rows["A.SZ"]["factor_a"] == 1.0
    assert rows["A.SZ"]["factor_b"] == 10.0
    assert rows["B.SH"]["factor_a"] is None  # +Inf is cleaned at the boundary
    assert rows["B.SH"]["factor_b"] is None  # left join keeps the universe row
    assert rows["C.BJ"]["factor_a"] is None
    assert rows["C.BJ"]["factor_b"] == 30.0
    assert "unused_a" not in result.columns
    assert "unused_b" not in result.columns
