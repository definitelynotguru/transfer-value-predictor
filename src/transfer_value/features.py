"""Leakage-safe, as-of transfer features. Pure functions over canonical tables.

Performance features only use league appearances with match_date < transfer_date from the
two latest completed source seasons before that date, plus the ongoing season's prefix.
Position prefers lineup positions from those same appearances; the current player position is
a flagged retrospective proxy and is exempt from the strict as-of claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from transfer_value.config import POSITIONS
from transfer_value.seasons import resolve_lookback

NUMERIC_FEATURES = [
    "goals",
    "assists",
    "minutes",
    "appearances",
    "goals_per90",
    "assists_per90",
    "age",
]
CARD_FEATURES = ["yellow_cards", "red_cards"]
CATEGORICAL_FEATURES = ["position"]

# Ordered: a candidate label row gets the first failing reason.
FEATURE_REASONS = [
    "no_lookback_coverage",
    "no_league_appearances_in_lookback",
    "insufficient_lookback_minutes",
    "invalid_per90",
    "missing_age",
    "missing_position",
]

FORBIDDEN_INPUT_TOKENS = ("fee", "market_value", "to_club", "destination", "transfer_id", "date")


def model_inputs(include_cards: bool) -> tuple[list[str], list[str]]:
    numeric = NUMERIC_FEATURES + (CARD_FEATURES if include_cards else [])
    return numeric, list(CATEGORICAL_FEATURES)


def assert_allowed_inputs(columns: list[str]) -> None:
    bad = [c for c in columns if any(tok in c for tok in FORBIDDEN_INPUT_TOKENS)]
    if bad:
        raise AssertionError(f"forbidden model inputs: {bad}")


def floor_age(dob: pd.Series, at: pd.Series) -> pd.Series:
    before_birthday = (at.dt.month < dob.dt.month) | (
        (at.dt.month == dob.dt.month) & (at.dt.day < dob.dt.day)
    )
    return (at.dt.year - dob.dt.year - before_birthday.astype(int)).astype("Float64")


def _lookback_appearances(
    candidates: pd.DataFrame, lookback: pd.DataFrame, appearances: pd.DataFrame
) -> pd.DataFrame:
    c = candidates[["transfer_id", "player_id", "transfer_date"]].merge(
        lookback, left_on="transfer_date", right_on="feature_cutoff_date", how="left"
    )
    m = c.merge(appearances, on="player_id", how="inner")
    in_season = (
        m["season_id"].eq(m["completed_1"]).fillna(False)
        | m["season_id"].eq(m["completed_2"]).fillna(False)
        | m["season_id"].eq(m["ongoing"]).fillna(False)
    )
    m = m[in_season & (m["match_date"] < m["transfer_date"]) & m["coverage_ok"]]
    return m.sort_values(["player_id", "match_date", "game_id", "appearance_id", "transfer_id"])


def _mode_position(apps: pd.DataFrame) -> pd.Series:
    """Most frequent lineup position; ties broken by fixed GK, DF, MF, FW order."""
    p = apps.dropna(subset=["lineup_position"])
    if p.empty:
        return pd.Series(dtype="string", name="position")
    counts = p.groupby(["transfer_id", "lineup_position"]).size().rename("n").reset_index()
    counts["order"] = counts["lineup_position"].map({k: i for i, k in enumerate(POSITIONS)})
    counts = counts.sort_values(["transfer_id", "n", "order"], ascending=[True, False, True])
    best = counts.drop_duplicates("transfer_id").set_index("transfer_id")["lineup_position"]
    return best.astype("string").rename("position")


def build_features(
    candidates: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    players: pd.DataFrame,
    seasons: pd.DataFrame,
    source_seasons: list[int],
    min_minutes: int,
    include_cards: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features for included rows, all candidates with feature_exclusion_reason).

    candidates needs transfer_id, player_id, transfer_date; other columns pass through.
    """
    if candidates["transfer_id"].duplicated().any():
        raise AssertionError("duplicate transfer_id in candidates")
    lookback = resolve_lookback(seasons, candidates["transfer_date"], source_seasons)
    apps = appearances.merge(lineups[["game_id", "player_id", "lineup_position"]], how="left")
    la = _lookback_appearances(candidates, lookback, apps)

    agg = la.groupby("transfer_id").agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes=("minutes", "sum"),
        appearances=("minutes", lambda m: int((m > 0).sum())),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        lookback_match_count=("game_id", "nunique"),
        first_match_date=("match_date", "min"),
        last_match_date=("match_date", "max"),
        lookback_season_ids=(
            "season_id",
            lambda s: ",".join(str(x) for x in sorted(s.unique())),
        ),
        lineup_position_rows=("lineup_position", "count"),
    )
    agg = agg.join(_mode_position(la))

    df = candidates.merge(
        lookback, left_on="transfer_date", right_on="feature_cutoff_date", how="left"
    )
    df = df.merge(agg, left_on="transfer_id", right_index=True, how="left")
    df = df.merge(
        players[["player_id", "name", "date_of_birth", "current_position"]],
        on="player_id",
        how="left",
    )

    df["lookback_minutes"] = df["minutes"]
    enough = df["minutes"] >= min_minutes
    df["goals_per90"] = (90 * df["goals"] / df["minutes"]).where(enough)
    df["assists_per90"] = (90 * df["assists"] / df["minutes"]).where(enough)
    df["age"] = floor_age(df["date_of_birth"], df["transfer_date"])
    df["age_is_approximate"] = False

    historical = df["position"].notna()
    df["position_source"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[historical, "position_source"] = "historical_appearance"
    use_proxy = ~historical & df["current_position"].notna()
    df.loc[use_proxy, "position"] = df.loc[use_proxy, "current_position"]
    df.loc[use_proxy, "position_source"] = "current_proxy"
    df["position_is_proxy"] = df["position_source"].eq("current_proxy").fillna(False)

    checks = [
        ~df["coverage_ok"].fillna(False).astype(bool),
        df["lookback_match_count"].isna(),
        ~enough.fillna(False),
        df["goals_per90"].isna() | df["assists_per90"].isna(),
        df["age"].isna(),
        df["position"].isna(),
    ]
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, mask in zip(FEATURE_REASONS, checks, strict=True):
        reason[reason.isna() & mask.astype(bool)] = name
    df["feature_exclusion_reason"] = reason

    included = df[reason.isna()].copy()
    int_cols = ["goals", "assists", "minutes", "appearances", "yellow_cards", "red_cards"]
    for c in int_cols + ["lookback_match_count", "lookback_minutes"]:
        included[c] = included[c].astype("int64")
    for c in NUMERIC_FEATURES:
        included[c] = included[c].astype(float)
    included = included.sort_values(["transfer_date", "transfer_id"]).reset_index(drop=True)
    if included["transfer_id"].duplicated().any():
        raise AssertionError("feature table is not one row per transfer")
    if not np.isfinite(included[NUMERIC_FEATURES].to_numpy()).all():
        raise AssertionError("non-finite model features")
    return included, df
