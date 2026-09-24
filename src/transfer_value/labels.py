"""Mode A label rules: paid permanent transfers only. Single owner of label inclusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered: each excluded row gets the first failing reason.
LABEL_REASONS = [
    "missing_player_id",
    "missing_transfer_date",
    "outside_study_window",
    "undisclosed_fee",
    "malformed_fee",
    "zero_fee_free_or_loan",
]


def label_exclusion_reason(
    transfers: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp
) -> pd.Series:
    """First failing label predicate per transfer, or <NA> when the row is a valid label.

    The source collapses free transfers, loans (including paid loans), and loan returns to a
    fee of 0, so they cannot be separated; all are excluded. Null fees are undisclosed or
    unknown and are never treated as 0.
    """
    d = transfers["transfer_date"]
    checks = [
        transfers["player_id"].isna(),
        d.isna(),
        (d < start) | (d >= end_exclusive),
        transfers["fee_class"] == "undisclosed",
        transfers["fee_class"] == "malformed",
        transfers["fee_class"] == "zero_fee",
    ]
    reason = pd.Series(pd.NA, index=transfers.index, dtype="string")
    for name, mask in zip(LABEL_REASONS, checks, strict=True):
        mask = mask.fillna(True).astype(bool)
        reason[reason.isna() & mask] = name
    ok = reason.isna()
    fee = transfers["fee_eur"]
    bad_fee = ok & ~(fee.notna() & (fee > 0) & np.isfinite(fee.astype(float)))
    if bad_fee.any():
        raise AssertionError("label rows with non-positive or non-finite fee")
    return reason


def apply_labels(
    transfers: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp
) -> tuple[pd.DataFrame, dict]:
    reason = label_exclusion_reason(transfers, start, end_exclusive)
    df = transfers.assign(label_exclusion_reason=reason)
    labeled = df[reason.isna()].copy()
    labeled["fee_log1p"] = np.log1p(labeled["fee_eur"].astype(float))

    in_window = df[~reason.isin(LABEL_REASONS[:3]).fillna(False)]
    stats = {
        "all_rows": {
            "total": len(df),
            "by_reason": reason.fillna("labeled").value_counts().sort_index().to_dict(),
        },
        "study_window": {
            "total": len(in_window),
            "paid_positive_fee": int(in_window["label_exclusion_reason"].isna().sum()),
            "undisclosed_null_fee": int((in_window["fee_class"] == "undisclosed").sum()),
            "zero_fee_free_or_loan_or_loan_end": int((in_window["fee_class"] == "zero_fee").sum()),
            "malformed_fee": int((in_window["fee_class"] == "malformed").sum()),
        },
        "notes": [
            "Free transfers, loans (paid or not) and loan returns share fee 0 in the source; "
            "they are excluded together and cannot be counted separately.",
            "Null fees are undisclosed or unknown and are excluded, never zero-filled.",
        ],
        "round_trip_diagnostic": round_trip_diagnostic(df, labeled),
    }
    return labeled.reset_index(drop=True), stats


def round_trip_diagnostic(all_transfers: pd.DataFrame, labeled: pd.DataFrame) -> dict:
    """Count labels followed by a zero/null-fee move back to the seller within 400 days.

    This flags possible loans or buy-backs for disclosure. It is not a filter.
    """
    back = all_transfers[all_transfers["fee_class"].isin(["zero_fee", "undisclosed"])]
    m = labeled[["transfer_id", "player_id", "transfer_date", "from_club_id", "to_club_id"]].merge(
        back[["player_id", "transfer_date", "from_club_id", "to_club_id"]],
        left_on=["player_id", "to_club_id", "from_club_id"],
        right_on=["player_id", "from_club_id", "to_club_id"],
        suffixes=("", "_back"),
    )
    gap = (m["transfer_date_back"] - m["transfer_date"]).dt.days
    flagged = m.loc[(gap > 0) & (gap <= 400), "transfer_id"].unique()
    return {"labels_flagged": len(flagged), "labels_total": len(labeled), "window_days": 400}
