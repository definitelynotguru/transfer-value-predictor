import pandas as pd
import pytest

from transfer_value.dataset import StudyWindowError, verify_study_window
from transfer_value.split import SplitError, chronological_folds, holdout


def _df(dates, cycles):
    return pd.DataFrame(
        {
            "transfer_id": [f"t{i}" for i in range(len(dates))],
            "player_id": [f"p{i}" for i in range(len(dates))],
            "transfer_date": pd.to_datetime(dates),
            "transfer_cycle": cycles,
        }
    )


def test_boundary_at_t_goes_to_test():
    df = _df(["2022-05-22", "2022-05-23", "2022-06-01"], [2021, 2022, 2022])
    train, test = holdout(df, pd.Timestamp("2022-05-23"), 1, 1)
    assert train["transfer_id"].tolist() == ["t0"]
    assert test["transfer_id"].tolist() == ["t1", "t2"]


def test_no_transfer_in_both_partitions(assembled, cfg):
    train, test = holdout(assembled["features"], pd.Timestamp(cfg.test_start), 1, 1)
    assert not set(train["transfer_id"]) & set(test["transfer_id"])
    assert (train["transfer_date"] < pd.Timestamp(cfg.test_start)).all()
    assert (test["transfer_date"] >= pd.Timestamp(cfg.test_start)).all()


def test_gates_refuse_small_data():
    df = _df(["2020-01-01", "2023-01-01"], [2020, 2023])
    with pytest.raises(SplitError, match="min_total_rows"):
        holdout(df, pd.Timestamp("2022-05-23"), 200, 1)
    with pytest.raises(SplitError, match="min_test_rows"):
        holdout(df, pd.Timestamp("2022-05-23"), 1, 40)


def test_duplicate_transfer_ids_rejected():
    df = _df(["2020-01-01", "2023-01-01"], [2020, 2023])
    df.loc[1, "transfer_id"] = "t0"
    with pytest.raises(SplitError, match="duplicate"):
        holdout(df, pd.Timestamp("2022-05-23"), 1, 1)


def test_folds_expand_and_only_validate_on_the_future():
    dates = [
        "2019-07-01",
        "2019-08-01",
        "2020-08-01",
        "2020-09-01",
        "2021-07-01",
        "2021-07-01",
        "2021-08-01",
    ]
    df = _df(dates, [2019, 2019, 2020, 2020, 2021, 2021, 2021])
    folds = chronological_folds(df, 2, 1, 1)
    assert [f.validation_cycle for f in folds] == [2020, 2021]
    assert folds[0].train_idx == [0, 1] and folds[0].val_idx == [2, 3]
    assert folds[1].train_idx == [0, 1, 2, 3] and folds[1].val_idx == [4, 5, 6]
    # rows sharing a date stay in one block
    assert {4, 5} <= set(folds[1].val_idx)


def test_folds_enforce_minimums():
    df = _df(["2019-07-01", "2020-08-01", "2021-07-01"], [2019, 2020, 2021])
    with pytest.raises(SplitError, match="below minimums"):
        chronological_folds(df, 2, 5, 1)
    with pytest.raises(SplitError, match="need"):
        chronological_folds(df, 3, 1, 1)


def test_pinned_dates_must_match_season_metadata(cfg, tables):
    verify_study_window(cfg, tables["seasons"])
    raw = {**cfg.raw, "split": {**cfg.raw["split"], "test_start": "2022-07-01"}}
    bad = type(cfg)(raw=raw, root=cfg.root)
    with pytest.raises(StudyWindowError, match="differ"):
        verify_study_window(bad, tables["seasons"])
