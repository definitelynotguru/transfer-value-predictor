"""Post-holdout follow-up: exposure normalization, price level, market-value comparator, CLI."""

import json
import math

import pandas as pd
import pytest
from typer.testing import CliRunner

from conftest import make_env
from transfer_value.cli import app
from transfer_value.features import model_inputs
from transfer_value.followup import (
    EXPOSURE_FEATURES,
    exposure_columns,
    league_price_level,
    market_value_asof,
    variants,
)
from transfer_value.io import sha256_file
from transfer_value.seasons import transfer_cycles

runner = CliRunner()


def _ts(s):
    return pd.Timestamp(s)


def test_winter_transfer_is_normalized_by_available_matches(assembled, tables):
    feats = assembled["features"]
    df = exposure_columns(feats, tables["games"], tables["appearances"], 4).set_index("player_id")
    winter = df.loc["205"]  # 2021-01-20: 2 of 4 games of 2020/21 played before D
    assert winter["window"] == "in_season"
    assert winter["ongoing_season_share"] == 0.5
    assert winter["lookback_season_equivalents"] == 2.5
    assert winter["available_team_matches"] == 10
    assert winter["minutes"] == 700 and winter["minutes_share"] == pytest.approx(700 / 900)
    assert winter["appearance_share"] == 1.0
    assert winter["goals_per_season"] == pytest.approx(20 / 2.5)

    summer = df.loc["101"]  # 2023-07-15: no season in progress
    assert summer["window"] == "off_season"
    assert summer["lookback_season_equivalents"] == 2.0
    assert summer["minutes_share"] == pytest.approx(720 / (90 * 8))


def test_price_level_uses_only_earlier_moves_touching_pl_clubs(tables):
    cycles = transfer_cycles(tables["seasons"])
    apps = pd.DataFrame({"season_id": [2020, 2020], "club_id": ["1", "1"]})
    moves = pd.DataFrame(
        {
            "transfer_date": [_ts("2020-12-01"), _ts("2021-01-05"), _ts("2021-01-10")],
            "from_club_id": ["1", "9", "1"],
            "to_club_id": ["2", "8", "2"],
            "fee_eur": [4e6, 99e6, 50e6],  # 2nd: no PL club; 3rd: on D itself
        }
    )
    out = league_price_level(pd.Series([_ts("2021-01-10")]), moves, apps, cycles, 365, 1)
    assert out.loc[0, "price_level_transfers"] == 1
    assert out.loc[0, "league_price_level"] == pytest.approx(math.log(4e6))
    with pytest.raises(ValueError, match="price level"):
        league_price_level(pd.Series([_ts("2021-01-10")]), moves, apps, cycles, 365, 2)


def test_market_value_is_strictly_before_and_fresh():
    rows = pd.DataFrame(
        {
            "transfer_id": ["a", "b", "c"],
            "player_id": ["1", "2", "3"],
            "transfer_date": [_ts("2023-07-15")] * 3,
        }
    )
    vals = pd.DataFrame(
        {
            "player_id": pd.Series(["1", "1", "1", "2"], dtype="string"),
            "valuation_date": pd.to_datetime(
                ["2023-06-01", "2023-07-15", "2023-08-01", "2021-01-01"]
            ).astype("datetime64[us]"),
            "market_value_eur": [10e6, 99e6, 99e6, 5e6],
        }
    ).sort_values("valuation_date")
    mv = market_value_asof(rows, vals, 365)
    assert mv.loc["a", "market_value_eur"] == 10e6  # same-day and later values ignored
    assert mv.loc["a", "valuation_age_days"] == 44
    assert mv.loc["b", "market_value_status"] == "stale" and pd.isna(
        mv.loc["b", "market_value_eur"]
    )
    assert mv.loc["c", "market_value_status"] == "no_prior_valuation"


def test_variants_start_with_headline_and_never_use_market_value():
    numeric, _ = model_inputs(False)
    vs = variants(numeric)
    assert vs[0] == {"name": "headline", "numeric": numeric}
    assert [v["name"] for v in vs] == [
        "headline",
        "headline+trend",
        "headline+price_level",
        "exposure",
        "exposure+trend",
        "exposure+price_level",
    ]
    for v in vs[3:]:
        assert set(EXPOSURE_FEATURES) <= set(v["numeric"])
        assert not {"goals", "assists", "minutes", "appearances"} & set(v["numeric"])
    assert all("market_value" not in c for v in vs for c in v["numeric"])


def test_followup_cli_leaves_headline_untouched(tmp_path_factory):
    root = tmp_path_factory.mktemp("followup")
    cfg = str(make_env(root))
    fup = str(root / "followup.yaml")
    r = runner.invoke(app, ["followup", "--config", fup])
    assert r.exit_code == 1  # no headline run yet

    assert runner.invoke(app, ["pipeline", "--config", cfg]).exit_code == 0
    art = root / "artifacts"
    headline = {p: sha256_file(art / p) for p in ("metrics.json", "test_predictions.parquet")}
    r = runner.invoke(app, ["followup", "--config", fup])
    assert r.exit_code == 0, r.output + str(r.exception)
    assert "not a headline result" in r.output
    assert {p: sha256_file(art / p) for p in headline} == headline

    out = json.loads((art / "followup" / "followup.json").read_text())
    assert sum(v["selected_by_cv"] for v in out["variants"].values()) == 1
    assert out["selected_variant"] in out["variant_order"]
    assert out["headline_metrics_sha256"] == headline["metrics.json"]
    head = json.loads((art / "metrics.json").read_text())
    assert out["variants"]["headline"]["test"]["mae_eur"] == pytest.approx(
        head["methods"][head["selected_model"]["family"]]["mae_eur"]
    )
    mv = out["market_value_comparator"]
    assert mv["status_counts"].get("stale") == 1  # Alex Example: only a stale prior value
    assert mv["status_counts"].get("no_prior_valuation") == 1  # Proxy Position
    assert mv["matched_rows"] == mv["test_rows"] - 2

    again = runner.invoke(app, ["followup", "--config", fup])
    assert again.exit_code == 0
    assert json.loads((art / "followup" / "followup.json").read_text()) == out
