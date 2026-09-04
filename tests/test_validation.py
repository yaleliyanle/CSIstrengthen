from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl

from ashare_lab.validation import Period, purge_period, validation_folds


def test_fixed_rolling_split_and_boundary_purge_are_small_and_explicit():
    cfg = SimpleNamespace(
        split={
            "method": "fixed",
            "train_start": "2020-01-01",
            "train_end": "2021-12-31",
            "validation_start": "2022-01-01",
            "validation_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2023-12-31",
            "live_start": "2024-01-01",
            "live_end": "2024-12-31",
        },
        rolling={
            "window": "expanding",
            "train_months": 12,
            "validation_months": 6,
            "step_months": 6,
        },
    )
    assert len(validation_folds(cfg)) == 1
    cfg.split["method"] = "rolling"
    assert len(validation_folds(cfg)) == 4

    calendar = [date(2024, 1, 1) + timedelta(days=index) for index in range(10)]
    panel = pl.DataFrame({"date": calendar[:5], "stock": ["A"] * 5})
    clean = purge_period(
        panel,
        calendar,
        Period("train", calendar[0], calendar[4]),
        horizon=2,
        execution="next_open",
    )
    assert clean["date"].to_list() == calendar[:2]
