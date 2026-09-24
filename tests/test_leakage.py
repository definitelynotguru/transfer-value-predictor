"""Mirrors the README worked example: Alex Example, recorded transfer date 2023-07-15."""

import pandas as pd

from transfer_value.features import NUMERIC_FEATURES, build_features, model_inputs
from transfer_value.model import Candidate, make_pipeline

PERF = ["goals", "assists", "minutes", "appearances", "goals_per90", "assists_per90"]
D = pd.Timestamp("2023-07-15")


def _features(tables, appearances):
    cand = pd.DataFrame({"transfer_id": ["alex"], "player_id": ["101"], "transfer_date": [D]})
    feats, _ = build_features(
        cand,
        appearances,
        tables["lineups"],
        tables["players"],
        tables["seasons"],
        list(range(2017, 2024)),
        90,
    )
    return feats.iloc[0]


def test_post_transfer_goals_are_ignored(tables):
    base = _features(tables, tables["appearances"])
    # The fixture already has 12 goals for Alex in 2023/24; none may count.
    assert base["goals"] == 8
    assert base["last_match_date"] < D
    assert base["lookback_season_ids"] == "2021,2022"


def test_same_day_and_later_appearances_leave_features_unchanged(tables):
    apps = tables["appearances"]
    base = _features(tables, apps)
    template = apps[apps["player_id"] == "101"].iloc[[0]]
    leaks = pd.concat(
        [
            template.assign(
                appearance_id="leak_same_day",
                game_id="leak1",
                match_date=D,
                season_id=2022,
                goals=5,
                minutes=90,
            ),
            template.assign(
                appearance_id="leak_later",
                game_id="leak2",
                match_date=pd.Timestamp("2023-08-20"),
                season_id=2022,
                goals=5,
                minutes=90,
            ),
        ]
    )
    leaky = _features(tables, pd.concat([apps, leaks], ignore_index=True))
    for col in PERF:
        assert leaky[col] == base[col], col


def test_market_value_and_fee_cannot_enter_the_model():
    numeric, categorical = model_inputs(False)
    assert "market_value_in_eur" not in numeric + categorical
    assert all("fee" not in c for c in numeric + categorical)
    try:
        make_pipeline(
            Candidate("linear"), NUMERIC_FEATURES + ["market_value_in_eur"], categorical, 42
        )
    except AssertionError:
        return
    raise AssertionError("market value was accepted as a model input")
