"""Deterministic micro-snapshot in the raw Transfermarkt schema.

Named players cover each rule the tests check; filler players give the pipeline enough rows
to train and evaluate. Seasons have 4 league games each; 2019/20 is extended into July 2020.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

SEASON_DATES = {
    2017: ["2017-08-12", "2017-11-01", "2018-02-01", "2018-05-13"],
    2018: ["2018-08-11", "2018-11-01", "2019-02-01", "2019-05-12"],
    2019: ["2019-08-10", "2019-11-01", "2020-02-01", "2020-07-26"],
    2020: ["2020-09-12", "2020-11-01", "2021-02-01", "2021-05-23"],
    2021: ["2021-08-14", "2021-11-01", "2022-02-01", "2022-05-22"],
    2022: ["2022-08-06", "2022-11-01", "2023-02-01", "2023-05-28"],
    2023: ["2023-08-12", "2023-11-01", "2024-02-01", "2024-05-19"],
}
CUP_GAME = {"game_id": 9001, "season": 2020, "date": "2021-01-09"}
# (game_id, competition, date, minutes, goals) for Alex Example (101, transfer 2023-07-15).
CONTEXT_APPS = [
    (9101, "CL", "2022-10-01", 80, 0),  # counted
    (9102, "CL", "2023-07-15", 90, 1),  # transfer day: excluded
    (9103, "L1", "2021-03-01", 90, 2),  # before the lookback window: excluded
    (9104, "L1", "2023-03-01", 70, 1),  # counted
    (9105, "FAC", "2022-01-08", 90, 1),  # cup: neither group
]
LINEUP_POS = ["Goalkeeper", "Centre-Back", "Central Midfield", "Centre-Forward"]
PLAYER_POS = ["Goalkeeper", "Defender", "Midfield", "Attack"]

FILLER_DATES = [
    "2019-07-01",
    "2019-08-01",
    "2020-01-15",
    "2020-08-20",
    "2020-10-01",
    "2021-01-20",
    "2021-07-01",
    "2021-08-20",
    "2022-07-01",
    "2022-08-15",
    "2023-01-20",
    "2023-07-01",
    "2023-08-20",
    "2024-01-15",
]


def _games() -> pd.DataFrame:
    rows, gid = [], 1000
    for season, dates in SEASON_DATES.items():
        for d in dates:
            gid += 1
            rows.append(
                {
                    "game_id": gid,
                    "competition_id": "GB1",
                    "season": season,
                    "date": d,
                    "home_club_id": 1,
                    "away_club_id": 2,
                    "home_club_goals": 1,
                    "away_club_goals": 0,
                }
            )
    for gid, comp, date, _, _ in CONTEXT_APPS:
        rows.append(
            {
                "game_id": gid,
                "competition_id": comp,
                "season": 2022,
                "date": date,
                "home_club_id": 1,
                "away_club_id": 5,
                "home_club_goals": 1,
                "away_club_goals": 1,
            }
        )
    rows.append(
        {
            **CUP_GAME,
            "competition_id": "FAC",
            "home_club_id": 1,
            "away_club_id": 3,
            "home_club_goals": 2,
            "away_club_goals": 2,
        }
    )
    return pd.DataFrame(rows)


def build(raw_dir: Path) -> None:
    games = _games()
    league = games[games["competition_id"] == "GB1"]
    apps, lineups, players, transfers = [], [], [], []

    def player(pid, name, dob, pos):
        players.append(
            {
                "player_id": pid,
                "name": name,
                "date_of_birth": f"{dob} 00:00:00",
                "position": pos,
                "last_season": 2023,
            }
        )

    def play(pid, seasons, goals=0, assists=0, minutes=90, lineup=None, only_first=None):
        g = league[league["season"].isin(seasons)]
        if only_first is not None:
            g = g.head(only_first)
        for r in g.itertuples():
            apps.append(
                {
                    "appearance_id": f"{r.game_id}_{pid}",
                    "game_id": r.game_id,
                    "player_id": pid,
                    "player_club_id": 1,
                    "player_current_club_id": 1,
                    "date": r.date,
                    "player_name": str(pid),
                    "competition_id": "GB1",
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "goals": goals,
                    "assists": assists,
                    "minutes_played": minutes,
                }
            )
            if lineup:
                lineups.append(
                    {
                        "game_lineups_id": f"l{r.game_id}_{pid}",
                        "date": r.date,
                        "game_id": r.game_id,
                        "player_id": pid,
                        "club_id": 1,
                        "player_name": str(pid),
                        "type": "starting_lineup",
                        "position": lineup,
                        "number": 9,
                        "team_captain": 0,
                    }
                )

    def transfer(pid, date, fee, frm=1, to=2):
        transfers.append(
            {
                "player_id": pid,
                "transfer_date": date,
                "transfer_season": "x",
                "from_club_id": frm,
                "to_club_id": to,
                "from_club_name": f"Club {frm}",
                "to_club_name": f"Club {to}",
                "transfer_fee": fee,
                "market_value_in_eur": 1e6,
                "player_name": str(pid),
            }
        )

    # 101 Alex Example: worked leakage row. Lookback 2021 + 2022 = 8 goals in 720 min.
    player(101, "Alex Example", "1998-07-16", "Attack")
    play(101, [2021, 2022], goals=1, lineup="Centre-Forward")
    play(101, [2023], goals=3, lineup="Centre-Forward")  # after the transfer: must be ignored
    transfer(101, "2023-07-15", 30_000_000)
    # Non-PL context for 101. Lookback date window is [2021-08-14, 2023-07-15).
    for gid, comp, date, mins, goals in CONTEXT_APPS:
        apps.append(
            {
                "appearance_id": f"{gid}_101",
                "game_id": gid,
                "player_id": 101,
                "player_club_id": 1,
                "player_current_club_id": 1,
                "date": date,
                "player_name": "101",
                "competition_id": comp,
                "yellow_cards": 0,
                "red_cards": 0,
                "goals": goals,
                "assists": 0,
                "minutes_played": mins,
            }
        )
    # 102 loan (fee 0), 103 undisclosed (null fee)
    player(102, "Loan Player", "1996-01-01", "Midfield")
    play(102, [2019, 2020], lineup="Central Midfield")
    transfer(102, "2021-01-10", 0.0)
    player(103, "Undisclosed Player", "1996-01-01", "Midfield")
    play(103, [2019, 2020], lineup="Central Midfield")
    transfer(103, "2021-02-01", None)
    # 104 insufficient minutes; 105 orphan with only a cup appearance
    player(104, "Low Minutes", "1997-01-01", "Defender")
    play(104, [2020], minutes=45, lineup="Centre-Back", only_first=1)
    transfer(104, "2021-07-05", 5_000_000)
    player(105, "Orphan Player", "1997-01-01", "Defender")
    apps.append(
        {
            "appearance_id": f"{CUP_GAME['game_id']}_105",
            "game_id": CUP_GAME["game_id"],
            "player_id": 105,
            "player_club_id": 1,
            "player_current_club_id": 1,
            "date": CUP_GAME["date"],
            "player_name": "105",
            "competition_id": "FAC",
            "yellow_cards": 0,
            "red_cards": 0,
            "goals": 2,
            "assists": 0,
            "minutes_played": 90,
        }
    )
    transfer(105, "2021-07-05", 6_000_000)
    # 106 summer gap: 2018 goals must be excluded; lookback is 2019 + 2020 only.
    player(106, "Summer Gap", "1995-05-05", "Midfield")
    play(106, [2018], goals=1, lineup="Central Midfield")
    play(106, [2019, 2020], lineup="Central Midfield")
    transfer(106, "2021-07-10", 12_000_000)
    # 107 extended 2019/20: mid-July transfer sees 2019 as ongoing; August sees it completed.
    player(107, "Extended Season", "1994-02-02", "Attack")
    play(107, [2017, 2018], lineup="Centre-Forward")
    play(107, [2019], goals=1, lineup="Centre-Forward")
    transfer(107, "2020-07-15", 8_000_000, 1, 3)
    transfer(107, "2020-08-10", 20_000_000, 3, 4)
    # 108 no lineup rows: current position proxy. 109 no position at all.
    player(108, "Proxy Position", "1996-03-03", "Defender")
    play(108, [2020, 2021])
    transfer(108, "2022-07-20", 9_000_000)
    player(109, "No Position", "1996-03-03", "Missing")
    play(109, [2020, 2021])
    transfer(109, "2022-07-21", 9_000_000)

    valuations = []

    def valuation(pid, date, value):
        valuations.append(
            {
                "player_id": pid,
                "date": date,
                "market_value_in_eur": value,
                "current_club_name": "Club 1",
                "current_club_id": 1,
                "player_club_domestic_competition_id": "GB1",
            }
        )

    # 101: same-day and later valuations must be ignored; the only earlier one is stale.
    valuation(101, "2022-01-01", 10_000_000)
    valuation(101, "2023-07-15", 99_000_000)
    valuation(101, "2023-08-01", 99_000_000)
    # 108 has no valuation at all.

    for i, d in enumerate(FILLER_DATES):
        pid = 200 + i
        player(pid, f"Filler {i}", f"{1990 + i % 8}-0{1 + i % 9}-15", PLAYER_POS[i % 4])
        play(
            pid,
            list(SEASON_DATES),
            goals=i % 3,
            assists=(i + 1) % 2,
            minutes=60 + 2 * i,
            lineup=LINEUP_POS[i % 4],
        )
        fee = 4_000_000 + 1_500_000 * i + 2_000_000 * (i % 3)
        transfer(pid, d, fee)
        valuation(pid, (pd.Timestamp(d) - pd.Timedelta(days=30)).date().isoformat(), 0.9 * fee)
    transfer(200, "2018-07-01", 3_000_000, 2, 1)  # outside the study window

    comps = pd.DataFrame(
        [
            {
                "competition_id": "GB1",
                "competition_code": "premier-league",
                "type": "domestic_league",
                "name": "premier-league",
            },
            {
                "competition_id": "FAC",
                "competition_code": "fa-cup",
                "type": "domestic_cup",
                "name": "fa-cup",
            },
            {
                "competition_id": "CL",
                "competition_code": "uefa-champions-league",
                "type": "international_cup",
                "name": "uefa-champions-league",
            },
            {
                "competition_id": "L1",
                "competition_code": "bundesliga",
                "type": "domestic_league",
                "name": "bundesliga",
            },
        ]
    )
    tables = {
        "competitions": comps,
        "games": games,
        "appearances": pd.DataFrame(apps),
        "game_lineups": pd.DataFrame(lineups),
        "players": pd.DataFrame(players),
        "transfers": pd.DataFrame(transfers),
        "player_valuations": pd.DataFrame(valuations),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        with gzip.GzipFile(raw_dir / f"{name}.csv.gz", "wb", mtime=0) as f:
            f.write(df.to_csv(index=False).encode())
