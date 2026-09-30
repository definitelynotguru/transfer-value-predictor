"""As-of context inputs from the raw all-competition tables. Follow-up only, never headline.

The headline features read Premier League appearances only. The raw snapshot also records
European club competitions, other first-tier leagues, and every PL result, so three context
groups can be built under the same as-of rule (match_date < transfer date):

- europe: minutes in the main UEFA club competitions inside the lookback date window.
- other_league: minutes and goals in first-tier leagues other than the study league inside
  the lookback date window (a recent arrival's earlier record, e.g. a Bundesliga season).
- team: mean league points the player's club took in the lookback PL matches he played.

The lookback date window runs from the first match of the earlier completed lookback season to
the day before the transfer, so it spans the same period as the PL lookback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CONTEXT_GROUPS = {
    "europe": ["europe_minutes"],
    "other_league": ["other_league_minutes", "other_league_goals"],
    "team": ["team_points_per_game"],
}
CONTEXT_FEATURES = [c for cols in CONTEXT_GROUPS.values() for c in cols]
RAW_CONTEXT_FILES = ("appearances.csv.gz", "games.csv.gz", "competitions.csv.gz")

_APP_COLS = ["player_id", "date", "competition_id", "goals", "minutes_played"]
_GAME_COLS = [
    "game_id",
    "competition_id",
    "home_club_id",
    "away_club_id",
    "home_club_goals",
    "away_club_goals",
]


def _read(path: Path, cols: list[str], ids: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in cols if c not in header]
    if missing:
        raise ValueError(f"{path.name} missing columns for context inputs: {missing}")
    df = pd.read_csv(path, usecols=cols, dtype={c: "string" for c in ids})
    for c in ids:
        df[c] = df[c].str.strip()
    return df


def load_raw_context(raw_dir: Path, player_ids: set[str]) -> dict[str, pd.DataFrame]:
    apps = _read(raw_dir / "appearances.csv.gz", _APP_COLS, ["player_id", "competition_id"])
    apps = apps[apps["player_id"].isin(player_ids)].copy()
    apps["match_date"] = pd.to_datetime(apps["date"], errors="coerce", format="ISO8601")
    apps["match_date"] = apps["match_date"].dt.normalize()
    for c in ("goals", "minutes_played"):
        apps[c] = pd.to_numeric(apps[c], errors="coerce")
    apps = apps.dropna(subset=["match_date", "goals", "minutes_played"])
    games = _read(
        raw_dir / "games.csv.gz",
        _GAME_COLS,
        ["game_id", "competition_id", "home_club_id", "away_club_id"],
    )
    comps = _read(raw_dir / "competitions.csv.gz", ["competition_id", "type"], ["competition_id"])
    return {"appearances": apps.drop(columns="date"), "games": games, "competitions": comps}


def club_points(league_apps: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """League points (3/1/0) the player's club took in each league appearance's game."""
    g = games.dropna(subset=["home_club_goals", "away_club_goals"])
    m = league_apps[["game_id", "club_id"]].merge(g, on="game_id", how="left", validate="m:1")
    home = m["club_id"].eq(m["home_club_id"]).to_numpy()
    away = m["club_id"].eq(m["away_club_id"]).to_numpy()
    if not (home | away)[m["home_club_goals"].notna().to_numpy()].all():
        raise AssertionError("appearance club is neither side of its game")
    gf = np.where(home, m["home_club_goals"], m["away_club_goals"]).astype(float)
    ga = np.where(home, m["away_club_goals"], m["home_club_goals"]).astype(float)
    pts = np.select([gf > ga, gf == ga], [3.0, 1.0], 0.0)
    pts[np.isnan(gf) | np.isnan(ga)] = np.nan
    return pd.Series(pts, index=league_apps.index)


def context_columns(
    feats: pd.DataFrame,
    seasons: pd.DataFrame,
    league_apps: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    study_league_ids: list[str],
    europe_ids: list[str],
) -> pd.DataFrame:
    """Add CONTEXT_FEATURES plus lookback_window_start. Only matches strictly before D count."""
    out = feats.copy()
    first_match = seasons.set_index("season_id")["first_match"]
    out["lookback_window_start"] = out["completed_2"].astype("Int64").map(first_match)
    if out["lookback_window_start"].isna().any():
        raise AssertionError("rows without a lookback window start")

    comps = raw["competitions"]
    other_leagues = set(comps.loc[comps["type"] == "domestic_league", "competition_id"]) - set(
        study_league_ids
    )
    keys = out[["transfer_id", "player_id", "transfer_date", "lookback_window_start"]]
    m = keys.merge(raw["appearances"], on="player_id", how="inner")
    m = m[(m["match_date"] < m["transfer_date"]) & (m["match_date"] >= m["lookback_window_start"])]

    def total(mask: pd.Series, col: str) -> pd.Series:
        return m[mask].groupby("transfer_id")[col].sum()

    europe = m["competition_id"].isin(europe_ids)
    other = m["competition_id"].isin(other_leagues)
    sums = {
        "europe_minutes": total(europe, "minutes_played"),
        "other_league_minutes": total(other, "minutes_played"),
        "other_league_goals": total(other, "goals"),
    }
    for col, s in sums.items():
        out[col] = out["transfer_id"].map(s).fillna(0.0).astype(float)

    la = league_apps[league_apps["minutes"] > 0].copy()
    la["points"] = club_points(la, raw["games"])
    lb = out[["transfer_id", "player_id", "transfer_date", "completed_1", "completed_2", "ongoing"]]
    lb = lb.merge(la[["player_id", "season_id", "match_date", "points"]], on="player_id")
    in_lookback = (
        lb["season_id"].eq(lb["completed_1"]).fillna(False)
        | lb["season_id"].eq(lb["completed_2"]).fillna(False)
        | lb["season_id"].eq(lb["ongoing"]).fillna(False)
    )
    lb = lb[in_lookback & (lb["match_date"] < lb["transfer_date"])]
    out["team_points_per_game"] = out["transfer_id"].map(lb.groupby("transfer_id")["points"].mean())
    if not np.isfinite(out[CONTEXT_FEATURES].to_numpy(dtype=float)).all():
        raise AssertionError("non-finite context inputs")
    return out
