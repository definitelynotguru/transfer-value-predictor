"""Hypothetical reported-fee estimate for one player at an as-of date, via the saved model."""

from __future__ import annotations

import datetime as dt
import unicodedata

import joblib
import pandas as pd

from transfer_value.dataset import load_interim
from transfer_value.evaluate import retransform
from transfer_value.features import build_features


class PredictError(ValueError):
    pass


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).casefold().split())


def resolve_player(players: pd.DataFrame, name: str | None, player_id: str | None) -> pd.Series:
    if player_id is not None:
        hit = players[players["player_id"] == str(player_id)]
        if hit.empty:
            raise PredictError(f"unknown player_id {player_id}")
        return hit.iloc[0]
    if not name:
        raise PredictError("pass --player or --player-id")
    exact = players[players["name"] == name]
    if exact.empty:
        exact = players[players["name"].fillna("").map(normalize_name) == normalize_name(name)]
    if exact.empty:
        raise PredictError(f"no player named {name!r}")
    if len(exact) > 1:
        ids = ", ".join(
            f"{r.player_id} ({r.name}, born {r.date_of_birth.date()})" for r in exact.itertuples()
        )
        raise PredictError(f"ambiguous name {name!r}; use --player-id with one of: {ids}")
    return exact.iloc[0]


def run_predict(cfg, player: str | None, player_id: str | None, date: str | None) -> dict:
    as_of_raw = date or cfg.study.get("prediction_as_of")
    if not as_of_raw:
        raise PredictError("no --date given and study.prediction_as_of is unset")
    as_of = pd.Timestamp(dt.date.fromisoformat(str(as_of_raw)))

    path = cfg.artifact_dir / "model.joblib"
    if not path.exists():
        raise PredictError(f"missing {path}; run train first")
    bundle = joblib.load(path)
    if bundle["config_hash"] != cfg.hash():
        raise PredictError("model was trained with a different config; rerun train")
    train_end = pd.Timestamp(bundle["train_date_range"][1])
    if as_of <= train_end:
        raise PredictError(
            f"as-of {as_of.date()} is not after the model's last training transfer "
            f"({train_end.date()}); the model would have seen the future"
        )

    tables = load_interim(cfg)
    coverage_end = tables["games"]["match_date"].max() + pd.Timedelta(days=1)
    if as_of > coverage_end:
        raise PredictError(
            f"as-of {as_of.date()} is beyond source coverage ({coverage_end.date()})"
        )

    p = resolve_player(tables["players"], player, player_id)
    cand = pd.DataFrame(
        {"transfer_id": ["prediction"], "player_id": [p["player_id"]], "transfer_date": [as_of]}
    )
    feats, all_rows = build_features(
        cand,
        tables["appearances"],
        tables["lineups"],
        tables["players"],
        tables["seasons"],
        [int(s) for s in cfg.study["source_seasons"]],
        cfg.min_minutes,
        bool(cfg.study["include_cards"]),
    )
    if feats.empty:
        reason = all_rows["feature_exclusion_reason"].iloc[0]
        raise PredictError(f"cannot build features for {p['name']} at {as_of.date()}: {reason}")
    row = feats.iloc[0]
    cols = bundle["numeric_features"] + bundle["categorical_features"]
    log_pred = bundle["pipeline"].predict(feats[cols])
    eur, _ = retransform(log_pred)
    return {
        "player": p["name"],
        "player_id": p["player_id"],
        "as_of": as_of.date().isoformat(),
        "position": row["position"],
        "position_source": row["position_source"],
        "lookback_seasons": row["lookback_season_ids"],
        "lookback_minutes": int(row["minutes"]),
        "goals": int(row["goals"]),
        "assists": int(row["assists"]),
        "age": int(row["age"]),
        "predicted_reported_fee_eur": float(eur[0]),
        "model": bundle["name"],
        "run_id": bundle["run_id"],
        "note": "Hypothetical estimate of a reported fee, not an observed fee or a valuation.",
    }
