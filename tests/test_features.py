import pandas as pd
import pytest

from transfer_value.features import assert_allowed_inputs, build_features, floor_age
from transfer_value.seasons import resolve_lookback


def _row(assembled, pid, date):
    c = assembled["candidates"]
    hit = c[(c["player_id"] == pid) & (c["transfer_date"] == pd.Timestamp(date))]
    assert len(hit) == 1
    return hit.iloc[0]


def test_per90_math(assembled):
    r = _row(assembled, "101", "2023-07-15")
    assert r["minutes"] == 720 and r["goals"] == 8
    assert r["goals_per90"] == pytest.approx(90 * 8 / 720)
    assert r["assists_per90"] == 0


def test_insufficient_minutes_dropped_not_zero_filled(assembled):
    r = _row(assembled, "104", "2021-07-05")
    assert r["minutes"] == 45
    assert pd.isna(r["goals_per90"])
    assert r["feature_exclusion_reason"] == "insufficient_lookback_minutes"


def test_orphan_with_only_cup_minutes_is_dropped(assembled):
    r = _row(assembled, "105", "2021-07-05")
    assert r["feature_exclusion_reason"] == "no_league_appearances_in_lookback"


def test_summer_gap_uses_two_latest_completed_seasons(assembled):
    r = _row(assembled, "106", "2021-07-10")
    assert r["lookback_season_ids"] == "2019,2020"
    assert pd.isna(r["ongoing"])
    assert r["goals"] == 0  # 2018 goals are outside the lookback
    assert r["minutes"] == 720


def test_extended_2019_20_is_ongoing_in_mid_july(assembled):
    r = _row(assembled, "107", "2020-07-15")
    assert (r["completed_1"], r["completed_2"], r["ongoing"]) == (2018, 2017, 2019)
    assert r["minutes"] == 11 * 90  # 8 completed-season games + 3 pre-D 2019/20 games
    assert r["goals"] == 3
    assert r["last_match_date"] == pd.Timestamp("2020-02-01")


def test_extended_2019_20_is_completed_by_august(assembled):
    r = _row(assembled, "107", "2020-08-10")
    assert (r["completed_1"], r["completed_2"]) == (2019, 2018)
    assert pd.isna(r["ongoing"])
    assert r["goals"] == 4
    assert r["last_match_date"] == pd.Timestamp("2020-07-26")


def test_season_whose_last_match_is_on_d_is_not_completed(tables):
    lb = resolve_lookback(
        tables["seasons"], pd.Series([pd.Timestamp("2020-07-26")]), list(range(2017, 2024))
    )
    assert lb.loc[0, "ongoing"] == 2019
    assert lb.loc[0, "completed_1"] == 2018


def test_position_proxy_is_flagged(assembled):
    r = _row(assembled, "108", "2022-07-20")
    assert r["position"] == "DF"
    assert r["position_source"] == "current_proxy"
    assert r["position_is_proxy"]
    h = _row(assembled, "101", "2023-07-15")
    assert h["position_source"] == "historical_appearance" and not h["position_is_proxy"]


def test_missing_position_is_dropped(assembled):
    assert _row(assembled, "109", "2022-07-21")["feature_exclusion_reason"] == "missing_position"


def test_floor_age_before_birthday():
    dob = pd.Series(pd.to_datetime(["1998-07-16", "1998-07-15"]))
    at = pd.Series(pd.to_datetime(["2023-07-15", "2023-07-15"]))
    assert floor_age(dob, at).tolist() == [24, 25]


def test_repeated_player_rows_are_separate_transfers(assembled):
    f = assembled["features"]
    rows = f[f["player_id"] == "107"]
    assert len(rows) == 2 and rows["transfer_id"].is_unique


def test_one_row_per_transfer_and_sorted(assembled):
    f = assembled["features"]
    assert f["transfer_id"].is_unique
    assert f["transfer_date"].is_monotonic_increasing


def test_forbidden_inputs_rejected():
    for bad in ("market_value_in_eur", "fee_log1p", "to_club_id", "transfer_date"):
        with pytest.raises(AssertionError):
            assert_allowed_inputs(["goals", bad])


def test_builder_rejects_duplicate_candidates(tables):
    cand = pd.DataFrame(
        {
            "transfer_id": ["a", "a"],
            "player_id": ["101", "101"],
            "transfer_date": [pd.Timestamp("2023-07-15")] * 2,
        }
    )
    with pytest.raises(AssertionError):
        build_features(
            cand,
            tables["appearances"],
            tables["lineups"],
            tables["players"],
            tables["seasons"],
            list(range(2017, 2024)),
            90,
        )
