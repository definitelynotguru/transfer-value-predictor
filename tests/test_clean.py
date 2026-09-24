import pandas as pd
import pytest

from transfer_value import clean
from transfer_value.config import ConfigError, load_config


def test_fee_parser_never_turns_missing_into_zero():
    raw = pd.Series(["1500000.0", "0.0", None, "", "-", "?", "abc", "-5", "inf"])
    fee, cls = clean.parse_fee(raw)
    assert cls.tolist() == [
        "paid",
        "zero_fee",
        "undisclosed",
        "undisclosed",
        "undisclosed",
        "undisclosed",
        "malformed",
        "malformed",
        "malformed",
    ]
    assert fee.iloc[0] == 1_500_000
    assert fee.iloc[1:].isna().all()


def test_league_filter_excludes_cup_appearances(tables):
    apps = tables["appearances"]
    assert set(apps["competition_id"]) == {"GB1"}
    assert "105" not in set(apps["player_id"])


def test_ids_are_strings_and_dates_are_date_only(tables):
    t = tables["transfers"]
    assert t["player_id"].dtype == "string"
    assert (t["transfer_date"].dropna() == t["transfer_date"].dropna().dt.normalize()).all()


def test_transfer_id_is_deterministic(tables):
    t = tables["transfers"]
    again = clean.transfer_key(t)
    assert (again == t["transfer_id"]).all()
    assert t["transfer_id"].is_unique


def test_conflicting_transfer_rows_fail():
    raw = pd.DataFrame(
        {
            "player_id": [1, 1],
            "transfer_date": ["2020-01-01"] * 2,
            "transfer_season": ["x"] * 2,
            "from_club_id": [1, 1],
            "to_club_id": [2, 2],
            "from_club_name": ["a"] * 2,
            "to_club_name": ["b"] * 2,
            "transfer_fee": [1e6, 2e6],
        }
    )
    with pytest.raises(clean.SchemaError, match="conflicting"):
        clean.clean_transfers(raw)


def test_exact_duplicate_transfers_are_removed_and_counted():
    row = {
        "player_id": 1,
        "transfer_date": "2020-01-01",
        "transfer_season": "x",
        "from_club_id": 1,
        "to_club_id": 2,
        "from_club_name": "a",
        "to_club_name": "b",
        "transfer_fee": 1e6,
    }
    df, stats = clean.clean_transfers(pd.DataFrame([row, row]))
    assert len(df) == 1 and stats["exact_duplicates_removed"] == 1


def test_duplicate_appearances_cannot_multiply_minutes(tables):
    apps = tables["appearances"]
    assert not apps.duplicated(["game_id", "player_id"]).any()


def test_missing_required_column_fails_loud(tmp_path):
    import gzip

    with gzip.open(tmp_path / "players.csv.gz", "wt") as f:
        f.write("player_id,name\n1,x\n")
    with pytest.raises(clean.SchemaError, match="missing required columns"):
        clean.read_raw(tmp_path, "players")


def test_competition_must_match_metadata():
    comps = pd.DataFrame(
        {"competition_id": ["GB1"], "competition_code": ["fa-cup"], "type": ["domestic_cup"]}
    )
    with pytest.raises(clean.SchemaError):
        clean.verify_competitions(comps, ["GB1"], "premier-league")


def test_production_config_rejects_weak_settings(tmp_path):
    import yaml

    base = yaml.safe_load(open("config.yaml"))
    base["study"]["min_minutes"] = 45
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(base))
    with pytest.raises(ConfigError, match="min_minutes"):
        load_config(p)
    base["study"]["min_minutes"] = 90
    base["cv"]["n_folds"] = 1
    p.write_text(yaml.safe_dump(base))
    with pytest.raises(ConfigError, match="n_folds"):
        load_config(p)
