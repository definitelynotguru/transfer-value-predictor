"""Normalize raw Transfermarkt tables into canonical, typed frames.

Raw column names stop here; the rest of the package sees canonical names only.
IDs are strings, dates are date-only (datetime64 at midnight, no timezone).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_value.config import RAW_FILES, SCHEMA

LINEUP_POSITION_MAP = {
    "Goalkeeper": "GK",
    "Centre-Back": "DF",
    "Left-Back": "DF",
    "Right-Back": "DF",
    "Sweeper": "DF",
    "Defender": "DF",
    "Defensive Midfield": "MF",
    "Central Midfield": "MF",
    "Attacking Midfield": "MF",
    "Left Midfield": "MF",
    "Right Midfield": "MF",
    "midfield": "MF",
    "Midfield": "MF",
    "Centre-Forward": "FW",
    "Second Striker": "FW",
    "Left Winger": "FW",
    "Right Winger": "FW",
    "Attack": "FW",
}

PLAYER_POSITION_MAP = {"Goalkeeper": "GK", "Defender": "DF", "Midfield": "MF", "Attack": "FW"}

ID_COLUMNS = {
    "player_id",
    "game_id",
    "appearance_id",
    "from_club_id",
    "to_club_id",
    "player_club_id",
    "competition_id",
}


class SchemaError(ValueError):
    pass


def read_raw(raw_dir: Path, table: str, extra: list[str] | None = None) -> pd.DataFrame:
    path = raw_dir / RAW_FILES[table]
    if not path.exists():
        raise FileNotFoundError(f"required raw file missing: {path}")
    header = pd.read_csv(path, nrows=0).columns
    wanted = SCHEMA[table] + [c for c in (extra or []) if c in header]
    missing = [c for c in SCHEMA[table] if c not in header]
    if missing:
        raise SchemaError(f"{path.name} missing required columns: {missing}")
    dtypes = {c: "string" for c in wanted if c in ID_COLUMNS}
    return pd.read_csv(path, usecols=wanted, dtype=dtypes)


def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="ISO8601").dt.normalize()


def _id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace("", pd.NA)


def transfer_key(df: pd.DataFrame) -> pd.Series:
    """Deterministic transfer_id from the natural key (the source has no stable ID)."""
    parts = (
        df["player_id"].fillna("")
        + "|"
        + df["transfer_date"].dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + df["from_club_id"].fillna("")
        + "|"
        + df["to_club_id"].fillna("")
    )
    return parts.map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:16])


def parse_fee(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (fee_eur, fee_class).

    fee_class: paid | zero_fee | undisclosed | malformed. Null/undisclosed is never 0.
    """
    text = raw.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    fee_class = pd.Series("paid", index=raw.index, dtype="string")
    fee_class[text.isna() | text.isin(["", "-", "?"])] = "undisclosed"
    bad = text.notna() & ~text.isin(["", "-", "?"]) & (numeric.isna() | ~np.isfinite(numeric))
    fee_class[bad | (numeric < 0)] = "malformed"
    fee_class[(fee_class == "paid") & (numeric == 0)] = "zero_fee"
    fee_eur = numeric.where(fee_class == "paid").astype("Float64")
    return fee_eur, fee_class


def clean_transfers(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(
        {
            "source_row": np.arange(len(raw)),
            "player_id": _id(raw["player_id"]),
            "transfer_date": parse_date(raw["transfer_date"]),
            "transfer_season": raw["transfer_season"].astype("string"),
            "from_club_id": _id(raw["from_club_id"]),
            "to_club_id": _id(raw["to_club_id"]),
            "from_club_name": raw["from_club_name"].astype("string"),
            "to_club_name": raw["to_club_name"].astype("string"),
            "raw_fee": raw["transfer_fee"].astype("string"),
        }
    )
    df["fee_eur"], df["fee_class"] = parse_fee(raw["transfer_fee"])
    df["transfer_id"] = transfer_key(df)

    natural = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
    compare = natural + ["raw_fee", "transfer_season"]
    exact_dup = df.duplicated(compare, keep="first")
    key_dup = df.duplicated(natural, keep=False) & ~df.duplicated(compare, keep=False)
    if key_dup.any():
        raise SchemaError(
            f"{int(key_dup.sum())} transfer rows share a natural key with conflicting fields"
        )
    stats = {"raw_rows": len(df), "exact_duplicates_removed": int(exact_dup.sum())}
    df = df[~exact_dup].sort_values(["player_id", "transfer_date", "transfer_id"])
    return df.reset_index(drop=True), stats


def clean_players(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "player_id": _id(raw["player_id"]),
            "name": raw["name"].astype("string"),
            "date_of_birth": parse_date(raw["date_of_birth"]),
            "current_position_raw": raw["position"].astype("string"),
        }
    )
    df["current_position"] = df["current_position_raw"].map(PLAYER_POSITION_MAP).astype("string")
    if df["player_id"].isna().any():
        raise SchemaError("players has null player_id")
    if df["player_id"].duplicated().any():
        raise SchemaError("players.player_id is not unique")
    return df.sort_values("player_id").reset_index(drop=True)


def clean_games(raw: pd.DataFrame, competition_ids: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "game_id": _id(raw["game_id"]),
            "competition_id": _id(raw["competition_id"]),
            "season_id": pd.to_numeric(raw["season"], errors="coerce").astype("Int64"),
            "match_date": parse_date(raw["date"]),
            "played": raw["home_club_goals"].notna() & raw["away_club_goals"].notna(),
        }
    )
    if df["game_id"].duplicated().any():
        raise SchemaError("games.game_id is not unique")
    df = df[df["competition_id"].isin(competition_ids)]
    if df[["season_id", "match_date"]].isna().any().any():
        raise SchemaError("league games with missing season or date")
    return df.sort_values(["match_date", "game_id"]).reset_index(drop=True)


def clean_appearances(raw: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """League appearances only, joined to game season. Fails on conflicting duplicates."""
    df = pd.DataFrame(
        {
            "appearance_id": _id(raw["appearance_id"]),
            "game_id": _id(raw["game_id"]),
            "player_id": _id(raw["player_id"]),
            "club_id": _id(raw["player_club_id"]),
            "appearance_date": parse_date(raw["date"]),
            "competition_id": _id(raw["competition_id"]),
            "minutes": pd.to_numeric(raw["minutes_played"], errors="coerce"),
            "goals": pd.to_numeric(raw["goals"], errors="coerce"),
            "assists": pd.to_numeric(raw["assists"], errors="coerce"),
            "yellow_cards": pd.to_numeric(raw["yellow_cards"], errors="coerce"),
            "red_cards": pd.to_numeric(raw["red_cards"], errors="coerce"),
        }
    )
    df = df[df["game_id"].isin(games["game_id"])]
    stat_cols = ["minutes", "goals", "assists", "yellow_cards", "red_cards"]
    exact_dup = df.duplicated(["game_id", "player_id"] + stat_cols, keep="first")
    df = df[~exact_dup]
    if df.duplicated(["game_id", "player_id"]).any():
        raise SchemaError("conflicting appearance rows for the same (game_id, player_id)")
    if df["appearance_id"].duplicated().any():
        raise SchemaError("appearance_id is not unique")

    invalid = df[stat_cols].isna().any(axis=1) | (df[stat_cols] < 0).any(axis=1)
    df = df[~invalid]
    df = df.merge(games[["game_id", "season_id", "match_date"]], on="game_id", how="left")
    date_mismatch = int((df["appearance_date"] != df["match_date"]).sum())
    df = df.drop(columns=["appearance_date"])
    stats = {
        "league_appearances": len(df),
        "exact_duplicates_removed": int(exact_dup.sum()),
        "invalid_stat_rows_removed": int(invalid.sum()),
        "appearance_vs_game_date_mismatch": date_mismatch,
    }
    order = ["player_id", "match_date", "game_id", "appearance_id"]
    return df.sort_values(order).reset_index(drop=True), stats


def clean_lineups(raw: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(
        {
            "game_id": _id(raw["game_id"]),
            "player_id": _id(raw["player_id"]),
            "lineup_position_raw": raw["position"].astype("string"),
        }
    )
    df = df[df["game_id"].isin(games["game_id"])].drop_duplicates()
    if df.duplicated(["game_id", "player_id"]).any():
        raise SchemaError("conflicting lineup positions for the same (game_id, player_id)")
    df["lineup_position"] = df["lineup_position_raw"].map(LINEUP_POSITION_MAP).astype("string")
    unmapped = df.loc[df["lineup_position"].isna(), "lineup_position_raw"]
    stats = {
        "league_lineup_rows": len(df),
        "unmapped_positions": unmapped.value_counts(dropna=False).to_dict(),
    }
    return df.sort_values(["game_id", "player_id"]).reset_index(drop=True), stats


def verify_competitions(raw: pd.DataFrame, competition_ids: list[str], code: str) -> None:
    comps = raw.assign(competition_id=_id(raw["competition_id"]))
    for cid in competition_ids:
        row = comps[comps["competition_id"] == cid]
        if row.empty:
            raise SchemaError(f"competition {cid} not in competitions table")
        r = row.iloc[0]
        if r["competition_code"] != code or r["type"] != "domestic_league":
            raise SchemaError(
                f"competition {cid} is {r['competition_code']}/{r['type']}, expected "
                f"{code}/domestic_league"
            )
