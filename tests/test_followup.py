"""Post-holdout follow-up: exposure normalization, price level, market-value comparator, CLI."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from conftest import make_env
from transfer_value import boosting
from transfer_value.charts import CHARTS
from transfer_value.cli import app
from transfer_value.conformal import conformal_quantile, coverage, interval_eur
from transfer_value.context import CONTEXT_FEATURES, context_columns, load_raw_context
from transfer_value.features import model_inputs
from transfer_value.followup import (
    EXPOSURE_FEATURES,
    context_variants,
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


def test_context_inputs_only_count_the_lookback_window_before_d(cfg, assembled, tables):
    feats = assembled["features"]
    raw = load_raw_context(cfg.raw_dir, set(feats["player_id"]))
    df = context_columns(
        feats, tables["seasons"], tables["appearances"], raw, ["GB1"], ["CL", "EL", "UCOL"]
    ).set_index("player_id")
    alex = df.loc["101"]  # transfer 2023-07-15, window from 2021-08-14
    assert alex["lookback_window_start"] == _ts("2021-08-14")
    assert alex["europe_minutes"] == 80  # the transfer-day CL match is excluded
    assert alex["other_league_minutes"] == 70  # the pre-window Bundesliga match is excluded
    assert alex["other_league_goals"] == 1  # FA Cup goals count in neither group
    assert alex["team_points_per_game"] == 3.0  # club 1 won every fixture league game
    others = df.drop(index="101")
    assert (others[["europe_minutes", "other_league_minutes"]] == 0).all().all()
    assert np.isfinite(df[CONTEXT_FEATURES].to_numpy()).all()


def test_context_variants_extend_every_first_round_set():
    numeric, _ = model_inputs(False)
    first = variants(numeric)
    second = context_variants(first)
    assert [v["name"] for v in second] == [f"{v['name']}+context" for v in first]
    for a, b in zip(first, second, strict=True):
        assert b["numeric"] == a["numeric"] + CONTEXT_FEATURES


def test_conformal_quantile_interval_and_coverage():
    scores = np.arange(1.0, 10.0)  # n = 9
    assert conformal_quantile(scores, 0.8) == 8.0  # ceil(10 * 0.8) = 8th smallest
    assert conformal_quantile(scores, 0.95) == math.inf  # ceil(9.5) = 10 > n
    lo, hi = interval_eur(np.log1p(np.array([10.0])), math.log(2))
    assert hi[0] == pytest.approx(21.0) and lo[0] == pytest.approx(4.5)  # (1 + fee) ×/÷ 2
    assert interval_eur(np.array([0.1]), 5.0)[0][0] == 0.0  # lower end clamped at zero
    c = coverage(np.array([0.0, 1.0, -1.0, 3.0]), np.zeros(4), 1.0)
    assert c == {
        "rows": 4,
        "coverage": 0.75,
        "share_above_upper": 0.25,
        "share_below_lower": 0.0,
    }


def test_monotone_boosting_respects_signs():
    numeric, categorical = model_inputs(False)
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({c: rng.uniform(0, 10, n) for c in numeric})
    X["position"] = rng.choice(["GK", "DF", "MF", "FW"], n)
    y = 15 + 0.3 * X["minutes"] - 0.2 * X["age"] + rng.normal(0, 1, n)
    params = boosting.grid()[0]
    pipe = boosting.make_gbm(params, numeric, categorical, True, 42).fit(X, y)
    probe = pd.concat([X.iloc[[0]]] * 11, ignore_index=True)
    probe["minutes"] = np.linspace(0, 10, 11)
    assert (np.diff(pipe.predict(probe)) >= 0).all()
    probe["age"] = np.linspace(0, 10, 11)
    probe["minutes"] = 5.0
    assert (np.diff(pipe.predict(probe)) <= 0).all()


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
    assert "enriched" in mv["methods"]

    ctx = out["context_round"]
    assert sum(v["selected_by_cv"] for v in ctx["variants"].values()) == 1
    assert ctx["selected_variant"].endswith("+context")
    conf = out["conformal"]
    for key in ("headline", "followup", "enriched"):
        assert set(conf[key]["levels"]) == {"0.80", "0.90"}
    h80 = conf["headline"]["levels"]["0.80"]
    assert h80["calibration_rows"] == sum(
        len(f["validation_transfer_ids"])
        for f in json.loads((art / "cv_results.json").read_text())["folds"]
    )
    assert {"gbm", "gbm_monotone"} <= set(out["boosting"])

    iv = json.loads((art / "followup" / "headline_interval.json").read_text())
    assert iv["level"] == 0.8 and iv["q_log"] == h80["q_log"]
    r = runner.invoke(app, ["predict", "--player", "Alex Example", "--config", cfg])
    assert r.exit_code == 0, r.output
    assert "80% training-set conformal interval" in r.output

    again = runner.invoke(app, ["followup", "--config", fup])
    assert again.exit_code == 0
    assert json.loads((art / "followup" / "followup.json").read_text()) == out

    r = runner.invoke(app, ["charts", "--config", fup])
    assert r.exit_code == 0, r.output + str(r.exception)
    manifest = json.loads((art / "charts" / "manifest.json").read_text())
    assert manifest["followup_sha256"] == sha256_file(art / "followup" / "followup.json")
    assert set(manifest["files"]) == set(CHARTS)
    assert {n: sha256_file(art / "charts" / n) for n in CHARTS} == manifest["files"]
    assert runner.invoke(app, ["charts", "--config", fup]).exit_code == 0
    assert json.loads((art / "charts" / "manifest.json").read_text()) == manifest

    (art / "metrics.json").write_text("{}")
    r = runner.invoke(app, ["charts", "--config", fup])
    assert r.exit_code == 1 and "does not match the headline manifest" in r.output
