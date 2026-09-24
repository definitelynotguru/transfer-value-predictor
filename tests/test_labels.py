import numpy as np
import pandas as pd

from transfer_value.labels import apply_labels, label_exclusion_reason


def _reasons(tables, cfg):
    t = tables["transfers"]
    r = label_exclusion_reason(
        t, pd.Timestamp(cfg.transfer_start), pd.Timestamp(cfg.transfer_end_exclusive)
    )
    return pd.Series(r.to_numpy(), index=t["player_id"] + "@" + t["transfer_date"].astype(str))


def test_loan_zero_fee_is_excluded(tables, cfg):
    assert _reasons(tables, cfg)["102@2021-01-10"] == "zero_fee_free_or_loan"


def test_null_fee_is_excluded_not_zero(tables, cfg):
    assert _reasons(tables, cfg)["103@2021-02-01"] == "undisclosed_fee"


def test_outside_window_is_excluded(tables, cfg):
    assert _reasons(tables, cfg)["200@2018-07-01"] == "outside_study_window"


def test_paid_permanent_is_included_with_log_target(tables, cfg):
    labeled, stats = apply_labels(
        tables["transfers"],
        pd.Timestamp(cfg.transfer_start),
        pd.Timestamp(cfg.transfer_end_exclusive),
    )
    row = labeled[labeled["player_id"] == "101"].iloc[0]
    assert row["fee_eur"] == 30_000_000
    assert np.isclose(row["fee_log1p"], np.log1p(30_000_000))
    assert (labeled["fee_eur"] > 0).all()
    assert labeled["transfer_id"].is_unique
    assert stats["study_window"]["zero_fee_free_or_loan_or_loan_end"] == 1
    assert stats["study_window"]["undisclosed_null_fee"] == 1


def test_every_excluded_row_has_exactly_one_reason(tables, cfg):
    r = _reasons(tables, cfg)
    labeled, _ = apply_labels(
        tables["transfers"],
        pd.Timestamp(cfg.transfer_start),
        pd.Timestamp(cfg.transfer_end_exclusive),
    )
    assert r.notna().sum() + len(labeled) == len(tables["transfers"])
