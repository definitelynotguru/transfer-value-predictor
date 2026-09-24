"""Single owner of every temporal partition: the T holdout and expanding chronological folds."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class Fold:
    index: int
    validation_cycle: int
    train_idx: list[int]
    val_idx: list[int]


def holdout(
    df: pd.DataFrame, test_start: pd.Timestamp, min_total: int, min_test: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train: transfer_date < T, test: transfer_date >= T."""
    if df["transfer_id"].duplicated().any():
        raise SplitError("duplicate transfer_id before split")
    train = df[df["transfer_date"] < test_start].copy()
    test = df[df["transfer_date"] >= test_start].copy()
    if len(df) < min_total:
        raise SplitError(f"{len(df)} supervised rows < min_total_rows {min_total}")
    if len(test) < min_test:
        raise SplitError(f"{len(test)} test rows < min_test_rows {min_test}")
    if train.empty:
        raise SplitError("empty training partition")
    if set(train["transfer_id"]) & set(test["transfer_id"]):
        raise SplitError("transfer_id present in both train and test")
    order = ["transfer_date", "transfer_id"]
    return (
        train.sort_values(order).reset_index(drop=True),
        test.sort_values(order).reset_index(drop=True),
    )


def chronological_folds(
    train: pd.DataFrame, n_folds: int, min_train: int, min_val: int
) -> list[Fold]:
    """Expanding folds; fold k validates on one transfer cycle and trains on all earlier cycles.

    Cycles are whole date blocks, so rows sharing a date never straddle a boundary, and no
    fold ever trains on a transfer later than the ones it validates.
    """
    if not train["transfer_date"].is_monotonic_increasing:
        raise SplitError("train must be sorted by transfer_date")
    cycles = sorted(train["transfer_cycle"].dropna().unique())
    if len(cycles) < n_folds + 1:
        raise SplitError(f"{len(cycles)} training cycles; need {n_folds + 1} for {n_folds} folds")
    folds = []
    for i, cyc in enumerate(cycles[-n_folds:]):
        tr = train.index[train["transfer_cycle"] < cyc].tolist()
        va = train.index[train["transfer_cycle"] == cyc].tolist()
        if len(tr) < min_train or len(va) < min_val:
            raise SplitError(
                f"fold {i} (cycle {cyc}): train {len(tr)} / validation {len(va)} below minimums "
                f"{min_train}/{min_val}"
            )
        if train.loc[tr, "transfer_date"].max() >= train.loc[va, "transfer_date"].min():
            raise SplitError(f"fold {i} trains on dates not strictly before validation")
        folds.append(Fold(i, int(cyc), tr, va))
    return folds


def split_summary(train: pd.DataFrame, test: pd.DataFrame, test_start: pd.Timestamp) -> dict:
    def side(d: pd.DataFrame) -> dict:
        return {
            "rows": len(d),
            "unique_players": int(d["player_id"].nunique()),
            "date_min": d["transfer_date"].min().date().isoformat(),
            "date_max": d["transfer_date"].max().date().isoformat(),
            "cycles": sorted(int(c) for c in d["transfer_cycle"].unique()),
            "position_proxy_rows": int(d["position_is_proxy"].sum()),
            "position_proxy_pct": round(100 * float(d["position_is_proxy"].mean()), 2),
        }

    rep = test["player_id"].value_counts()
    return {
        "test_start": test_start.date().isoformat(),
        "train": side(train),
        "test": side(test),
        "test_players_with_repeat_transfers": int((rep > 1).sum()),
        "players_in_both_train_and_test": len(set(train["player_id"]) & set(test["player_id"])),
    }
