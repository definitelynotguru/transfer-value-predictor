"""Competition-season metadata, completion, transfer cycles, and lookback resolution.

Season identity comes from the source season IDs. A season is complete only when every
scheduled game is present and played; completion date is the last match date. No July-June
calendar buckets are used anywhere.
"""

from __future__ import annotations

import pandas as pd


def season_table(games: pd.DataFrame, games_per_season: int) -> pd.DataFrame:
    g = games.groupby("season_id", sort=True).agg(
        first_match=("match_date", "min"),
        last_match=("match_date", "max"),
        n_games=("game_id", "nunique"),
        n_played=("played", "sum"),
    )
    g["expected_games"] = games_per_season
    g["complete"] = (g["n_games"] == games_per_season) & (g["n_played"] == games_per_season)
    return g.reset_index()


def transfer_cycles(seasons: pd.DataFrame) -> pd.DataFrame:
    """Cycle s = [day after season s-1's last match, day after season s's last match)."""
    s = seasons.sort_values("season_id").reset_index(drop=True)
    s["cycle_end_exclusive"] = s["last_match"] + pd.Timedelta(days=1)
    s["cycle_start"] = s["cycle_end_exclusive"].shift(1)
    return s[["season_id", "cycle_start", "cycle_end_exclusive"]].dropna().reset_index(drop=True)


def cycle_of(dates: pd.Series, cycles: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NA, index=dates.index, dtype="Int64")
    for c in cycles.itertuples():
        out[(dates >= c.cycle_start) & (dates < c.cycle_end_exclusive)] = c.season_id
    return out


def resolve_lookback(
    seasons: pd.DataFrame, dates: pd.Series, source_seasons: list[int]
) -> pd.DataFrame:
    """For each distinct date D: two latest seasons completed before D, plus any ongoing season.

    Completed before D means complete and last_match < D. Ongoing means first_match < D <=
    last_match (a season whose last match falls on D is still ongoing; its D match is excluded
    later by match_date < D). coverage_ok requires both completed seasons to be in source_seasons.
    """
    allowed = set(source_seasons)
    s = seasons[seasons["season_id"].isin(allowed)].sort_values("season_id")
    rows = []
    for d in sorted(pd.Series(dates).dropna().unique()):
        d = pd.Timestamp(d)
        done = s[s["complete"] & (s["last_match"] < d)]["season_id"].tolist()[-2:]
        ongoing = s[(s["first_match"] < d) & (s["last_match"] >= d)]["season_id"].tolist()
        rows.append(
            {
                "feature_cutoff_date": d,
                "completed_1": done[-1] if len(done) >= 1 else pd.NA,
                "completed_2": done[-2] if len(done) >= 2 else pd.NA,
                "ongoing": ongoing[0] if ongoing else pd.NA,
                "coverage_ok": len(done) == 2,
            }
        )
    out = pd.DataFrame(
        rows,
        columns=["feature_cutoff_date", "completed_1", "completed_2", "ongoing", "coverage_ok"],
    )
    for c in ("completed_1", "completed_2", "ongoing"):
        out[c] = out[c].astype("Int64")
    out["coverage_ok"] = out["coverage_ok"].astype(bool)
    return out
