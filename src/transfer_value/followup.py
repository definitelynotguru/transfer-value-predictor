"""Post-holdout follow-up: exposure-normalized inputs, time terms, and a market-value comparator.

Everything here was designed after the headline holdout had been scored, so none of it is a
headline result. The headline artifacts are read, verified, and never rewritten. Variant
selection still uses only chronological CV inside the training window; the test set is scored
once per variant afterwards and reported as a labelled follow-up.

Transfermarkt market value enters only as a comparison row. It is never a model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from transfer_value.config import Config, config_hash, load_config
from transfer_value.dataset import load_interim
from transfer_value.evaluate import metrics, paired_cluster_bootstrap, retransform
from transfer_value.features import assert_allowed_inputs, model_inputs
from transfer_value.io import (
    frame_fingerprint,
    read_json,
    read_parquet,
    sha256_file,
    write_json,
    write_parquet,
)
from transfer_value.model import (
    Candidate,
    coefficient_table,
    cross_validate,
    fit_checked,
    make_pipeline,
    rank_key,
)
from transfer_value.seasons import cycle_of, transfer_cycles
from transfer_value.source import verify_raw_files
from transfer_value.split import chronological_folds, holdout

RAW_TOTALS = ("goals", "assists", "minutes", "appearances")
EXPOSURE_FEATURES = ["goals_per_season", "assists_per_season", "minutes_share", "appearance_share"]
TIME_TERMS = {"none": [], "trend": ["cycle_trend"], "price_level": ["league_price_level"]}
VALUATION_COLUMNS = ["player_id", "date", "market_value_in_eur"]


@dataclass(frozen=True)
class FollowupConfig:
    raw: dict[str, Any]
    root: Path

    @property
    def headline(self) -> Config:
        return load_config(self.root / self.raw["headline_config"])

    @property
    def raw_dir(self) -> Path:
        return self.root / self.raw["data"]["raw_dir"]

    @property
    def output_dir(self) -> Path:
        return self.root / self.raw["data"]["output_dir"]

    def hash(self) -> str:
        return config_hash(self.raw)


def load_followup_config(path: str | Path) -> FollowupConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"follow-up config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    for key in ("headline_config", "source", "data", "price_level", "market_value"):
        if key not in raw:
            raise ValueError(f"follow-up config missing {key}")
    return FollowupConfig(raw=raw, root=p.resolve().parent)


def variants(numeric: list[str]) -> list[dict]:
    """Fixed order; also the variant tie-breaker. The unmodified headline inputs come first."""
    exposure = EXPOSURE_FEATURES + [c for c in numeric if c not in RAW_TOTALS]
    out = []
    for base, cols in (("headline", numeric), ("exposure", exposure)):
        for term, extra in TIME_TERMS.items():
            name = base if term == "none" else f"{base}+{term}"
            out.append({"name": name, "numeric": cols + extra})
    for v in out:
        assert_allowed_inputs(v["numeric"])
    return out


def exposure_columns(
    feats: pd.DataFrame, games: pd.DataFrame, appearances: pd.DataFrame, games_per_season: int
) -> pd.DataFrame:
    """Normalize lookback totals by how much league football was available before D.

    A completed lookback season counts as one season; an ongoing season counts as the share of
    its scheduled games played before D. Winter transfers otherwise carry an extra half season.
    """
    out = feats.copy()
    frac = pd.Series(0.0, index=out.index)
    ongoing = out["ongoing"].notna()
    for season, idx in out[ongoing].groupby("ongoing").groups.items():
        dates = np.sort(games.loc[games["season_id"] == season, "match_date"].to_numpy())
        before = np.searchsorted(dates, out.loc[idx, "transfer_date"].to_numpy(), side="left")
        frac[idx] = before / games_per_season
    team_matches = (
        appearances.groupby(["season_id", "club_id"])["game_id"]
        .nunique()
        .groupby("season_id")
        .max()
    )

    def matches(col: str) -> pd.Series:
        return out[col].map(team_matches).astype(float).fillna(0.0)

    completed = out["completed_1"].notna().astype(int) + out["completed_2"].notna().astype(int)
    available = matches("completed_1") + matches("completed_2") + matches("ongoing") * frac
    out["ongoing_season_share"] = frac
    out["lookback_season_equivalents"] = completed + frac
    out["available_team_matches"] = available
    out["goals_per_season"] = out["goals"] / out["lookback_season_equivalents"]
    out["assists_per_season"] = out["assists"] / out["lookback_season_equivalents"]
    out["minutes_share"] = out["minutes"] / (90 * available)
    out["appearance_share"] = out["appearances"] / available
    out["window"] = np.where(ongoing, "in_season", "off_season")
    if (available <= 0).any():
        raise AssertionError("rows with no available team matches")
    return out


def league_price_level(
    dates: pd.Series,
    transfers: pd.DataFrame,
    appearances: pd.DataFrame,
    cycles: pd.DataFrame,
    window_days: int,
    min_transfers: int,
) -> pd.DataFrame:
    """Trailing median positive fee of moves touching a PL club, strictly before each date.

    A move counts when its buying or selling club played in the PL in the move's cycle season.
    No row's own fee can enter its level: the window ends the day before D.
    """
    pl = appearances[["season_id", "club_id"]].drop_duplicates()
    pl = set(zip(pl["season_id"].astype(int), pl["club_id"], strict=True))
    t = transfers[transfers["fee_eur"].fillna(0) > 0].copy()
    t["cycle"] = cycle_of(t["transfer_date"], cycles)
    t = t[t["cycle"].notna()]
    touches = [
        (c, f) in pl or (c, to) in pl
        for c, f, to in zip(t["cycle"].astype(int), t["from_club_id"], t["to_club_id"], strict=True)
    ]
    t = t[touches].sort_values("transfer_date")
    t_dates = t["transfer_date"].to_numpy()
    fees = t["fee_eur"].to_numpy(dtype=float)
    rows = []
    for d in sorted(pd.Series(dates).unique()):
        d = pd.Timestamp(d)
        lo = np.searchsorted(t_dates, (d - pd.Timedelta(days=window_days)).to_datetime64(), "left")
        hi = np.searchsorted(t_dates, d.to_datetime64(), "left")
        n = int(hi - lo)
        if n < min_transfers:
            raise ValueError(f"price level for {d.date()} has {n} < {min_transfers} transfers")
        rows.append(
            {
                "transfer_date": d,
                "league_price_level": float(np.log(np.median(fees[lo:hi]))),
                "price_level_transfers": n,
            }
        )
    return pd.DataFrame(rows)


def clean_valuations(raw: pd.DataFrame) -> pd.DataFrame:
    v = pd.DataFrame(
        {
            "player_id": raw["player_id"].astype("string").str.strip(),
            "valuation_date": pd.to_datetime(raw["date"], errors="coerce").dt.normalize(),
            "market_value_eur": pd.to_numeric(raw["market_value_in_eur"], errors="coerce"),
        }
    ).dropna()
    v = v[v["market_value_eur"] > 0]
    v["valuation_date"] = v["valuation_date"].astype("datetime64[us]")
    return v.sort_values(["valuation_date", "player_id"]).reset_index(drop=True)


def market_value_asof(rows: pd.DataFrame, valuations: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """Latest valuation strictly before the transfer date, within max_days; else NaN."""
    left = rows[["transfer_id", "player_id", "transfer_date"]].copy()
    left["player_id"] = left["player_id"].astype("string")
    left["transfer_date"] = left["transfer_date"].astype("datetime64[us]")
    m = pd.merge_asof(
        left.sort_values("transfer_date"),
        valuations,
        left_on="transfer_date",
        right_on="valuation_date",
        by="player_id",
        allow_exact_matches=False,
        direction="backward",
    )
    m["valuation_age_days"] = (m["transfer_date"] - m["valuation_date"]).dt.days
    stale = m["valuation_age_days"] > max_days
    m.loc[stale, "market_value_eur"] = np.nan
    m["market_value_status"] = np.select(
        [m["valuation_date"].isna(), stale], ["no_prior_valuation", "stale"], "matched"
    )
    return m.set_index("transfer_id")[
        ["market_value_eur", "valuation_date", "valuation_age_days", "market_value_status"]
    ]


def _verify_headline(fcfg: FollowupConfig, cfg: Config) -> dict:
    art = cfg.artifact_dir
    manifest = read_json(art / "manifest.json")
    fmeta = read_json(cfg.processed_dir / "features_manifest.json")
    pinned = fcfg.raw.get("headline_run_id")
    if pinned and manifest["run_id"] != pinned:
        raise RuntimeError(f"headline run {manifest['run_id']} != pinned {pinned}")
    if manifest["run_id"] != fmeta["run_id"]:
        raise RuntimeError("headline manifest and features manifest disagree; rerun pipeline")
    for rel in ("metrics.json", "test_predictions.parquet", "cv_results.json"):
        if sha256_file(art / rel) != manifest["artifact_sha256"][rel]:
            raise RuntimeError(f"artifacts/{rel} does not match the headline manifest")
    return manifest


def _valuations(fcfg: FollowupConfig) -> tuple[pd.DataFrame, str]:
    name = fcfg.raw["market_value"]["file"]
    digest = verify_raw_files(fcfg.raw_dir, [name])[name]
    pin = (fcfg.raw["source"].get("sha256") or {}).get(name)
    if pin and pin != digest:
        raise ValueError(f"{name}: sha256 {digest} != follow-up pin {pin}")
    raw = pd.read_csv(fcfg.raw_dir / name, usecols=VALUATION_COLUMNS, dtype={"player_id": str})
    return clean_valuations(raw), digest


def _fit_predict(c: Candidate, numeric, categorical, train, test, rs):
    pipe, issues = fit_checked(
        make_pipeline(c, numeric, categorical, rs),
        train[numeric + categorical],
        train["fee_log1p"].to_numpy(),
    )
    if issues:
        raise RuntimeError(f"final fit of {c.name} had issues: {issues}")
    return pipe, pipe.predict(test[numeric + categorical])


def _by_window(frame: pd.DataFrame, log_res: dict[str, pd.Series], abs_err=None) -> dict:
    out = {}
    for w, g in frame.groupby("window"):
        row = {
            "rows": len(g),
            "mean_raw_minutes": float(g["minutes"].mean()),
            "mean_season_equivalents": float(g["lookback_season_equivalents"].mean()),
            "mean_minutes_share": float(g["minutes_share"].mean()),
        }
        for k, s in log_res.items():
            row[f"mean_log_residual__{k}"] = float(s.loc[g.index].mean())
        for k, s in (abs_err or {}).items():
            row[f"mae_eur__{k}"] = float(s.loc[g.index].mean())
        out[str(w)] = row
    return out


def run_followup(fcfg: FollowupConfig) -> dict:
    cfg = fcfg.headline
    cfg.require_pinned()
    head_manifest = _verify_headline(fcfg, cfg)
    feats = read_parquet(cfg.processed_dir / "features.parquet")
    if frame_fingerprint(feats) != head_manifest["features_fingerprint"]:
        raise RuntimeError("features.parquet does not match the headline run")
    head_cv = read_json(cfg.artifact_dir / "cv_results.json")
    head_pred = read_parquet(cfg.artifact_dir / "test_predictions.parquet")
    tables = load_interim(cfg)
    gps = int(cfg.raw["competition"]["games_per_season"])
    pl_cfg = fcfg.raw["price_level"]

    df = exposure_columns(feats, tables["games"], tables["appearances"], gps)
    df["cycle_trend"] = df["transfer_cycle"].astype(float)
    level = league_price_level(
        df["transfer_date"],
        tables["transfers"],
        tables["appearances"],
        transfer_cycles(tables["seasons"]),
        int(pl_cfg["window_days"]),
        int(pl_cfg["min_transfers"]),
    )
    df = df.merge(level, on="transfer_date", how="left", validate="many_to_one")

    rs = int(cfg.runtime["random_state"])
    numeric, categorical = model_inputs(bool(cfg.study["include_cards"]))
    train, test = holdout(
        df, pd.Timestamp(cfg.test_start), cfg.split["min_total_rows"], cfg.split["min_test_rows"]
    )
    folds = chronological_folds(
        train, cfg.cv["n_folds"], cfg.cv["min_train_rows"], cfg.cv["min_validation_rows"]
    )

    vs = variants(numeric)
    cv_rows = []
    for vi, v in enumerate(vs):
        results = [r for r in cross_validate(train, folds, v["numeric"], categorical, rs)]
        best = min((r for r in results if r["eligible"]), key=rank_key)
        cv_rows.append({"variant": v["name"], "variant_order": vi, "best": best, "all": results})

    head_row = cv_rows[0]["best"]
    if head_row["name"] != head_cv["selected"] or not np.isclose(
        head_row["mean_log_mae"],
        next(c["mean_log_mae"] for c in head_cv["candidates"] if c["name"] == head_cv["selected"]),
    ):
        raise AssertionError("headline variant does not reproduce the headline CV selection")

    def overall_key(row):
        m, s, order = rank_key(row["best"])
        return (m, s, row["variant_order"], order)

    chosen = min(cv_rows, key=overall_key)

    y = test["fee_eur"].to_numpy(dtype=float)
    y_log_train = train["fee_log1p"]
    per_variant, preds_log, preds_eur, in_sample, pipes = {}, {}, {}, {}, {}
    for row, v in zip(cv_rows, vs, strict=True):
        c = Candidate(row["best"]["family"], row["best"]["params"])
        pipe, pl = _fit_predict(c, v["numeric"], categorical, train, test, rs)
        eur, clamped = retransform(pl)
        preds_log[v["name"]], preds_eur[v["name"]], pipes[v["name"]] = pl, eur, pipe
        in_sample[v["name"]] = y_log_train - pipe.predict(train[v["numeric"] + categorical])
        per_variant[v["name"]] = {
            "model": c.name,
            "numeric_features": v["numeric"],
            "cv_mean_log_mae": row["best"]["mean_log_mae"],
            "cv_std_log_mae": row["best"]["std_log_mae"],
            "cv_fold_log_mae": row["best"]["fold_log_mae"],
            "selected_by_cv": row is chosen,
            "clamped_at_zero": clamped,
            "test": metrics(y, eur, pl),
            "test_mean_log_residual": float(np.mean(np.log1p(y) - pl)),
        }

    hp = head_pred.set_index("transfer_id").loc[test["transfer_id"]]
    if not np.allclose(hp["prediction_log"].to_numpy(), preds_log["headline"]):
        raise AssertionError("headline variant does not reproduce the headline test predictions")

    sel = chosen["variant"]
    ae = {
        "followup": np.abs(y - preds_eur[sel]),
        "headline": np.abs(y - preds_eur["headline"]),
    }
    rt = cfg.runtime
    boot_args = (
        int(rt["bootstrap_replicates"]),
        int(rt["bootstrap_seed"]),
        float(rt["bootstrap_confidence"]),
    )
    boot = paired_cluster_bootstrap(
        test["player_id"], ae, "followup", *boot_args, names=("follow-up", "headline")
    )

    idx_test = test.index
    log_res_test = {
        "headline": pd.Series(np.log1p(y) - preds_log["headline"], index=idx_test),
        "followup": pd.Series(np.log1p(y) - preds_log[sel], index=idx_test),
    }
    abs_err_test = {k: pd.Series(v, index=idx_test) for k, v in ae.items()}
    window = {
        "definition": "in_season: a league season was in progress at the transfer date, so the "
        "lookback includes a partial ongoing season on top of two completed ones",
        "test": _by_window(test, log_res_test, abs_err_test),
        "train_in_sample": _by_window(
            train, {"headline": in_sample["headline"], "followup": in_sample[sel]}
        ),
    }

    valuations, val_digest = _valuations(fcfg)
    mv_cfg = fcfg.raw["market_value"]
    mv = market_value_asof(test, valuations, int(mv_cfg["max_staleness_days"])).loc[
        test["transfer_id"]
    ]
    matched = mv["market_value_status"].eq("matched").to_numpy()
    train_median = float(train["fee_eur"].median())
    ym = y[matched]
    comp_preds = {
        "headline": preds_eur["headline"][matched],
        "followup": preds_eur[sel][matched],
        "train_median": np.full(matched.sum(), train_median),
        "market_value": mv["market_value_eur"].to_numpy(dtype=float)[matched],
    }
    comp_metrics = {k: metrics(ym, p, np.log1p(p)) for k, p in comp_preds.items()}
    comp_ae = {k: np.abs(ym - p) for k, p in comp_preds.items()}
    players = test["player_id"].to_numpy()[matched]
    mv_boot = {
        ref: paired_cluster_bootstrap(
            pd.Series(players),
            {ref: comp_ae[ref], "market_value": comp_ae["market_value"]},
            ref,
            *boot_args,
            names=(ref, "market value"),
        )
        for ref in ("headline", "followup")
    }
    ages = mv["valuation_age_days"].to_numpy()[matched]
    market_value = {
        "note": "Comparator only. Transfermarkt market value is never a model input. It is an "
        "expert estimate that may already reflect transfer rumours, so it is not a ceiling and "
        "beating or losing to it proves neither leakage nor purity.",
        "rule": f"latest valuation with valuation_date < transfer_date and at most "
        f"{int(mv_cfg['max_staleness_days'])} days old",
        "test_rows": len(test),
        "matched_rows": int(matched.sum()),
        "status_counts": {
            str(k): int(v) for k, v in mv["market_value_status"].value_counts().items()
        },
        "median_valuation_age_days": float(np.median(ages)) if len(ages) else None,
        "methods": comp_metrics,
        "bootstrap": mv_boot,
        "share_fee_above_market_value": float(np.mean(ym > comp_preds["market_value"])),
    }

    coefs = coefficient_table(pipes[sel], per_variant[sel]["model"], True)
    out_pred = test[
        ["transfer_id", "player_id", "name", "transfer_date", "window", "fee_eur"]
    ].copy()
    out_pred["fee_eur"] = out_pred["fee_eur"].astype(float)
    for k in ("headline", sel):
        out_pred[f"pred_eur__{k}"] = preds_eur[k]
    out_pred["market_value_eur"] = mv["market_value_eur"].to_numpy()
    out_pred["valuation_age_days"] = mv["valuation_age_days"].to_numpy()
    fcfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(out_pred, fcfg.output_dir / "test_predictions.parquet")

    result = {
        "note": "Follow-up designed after the headline holdout was scored. Not a headline "
        "result. Variant and model selection used training-window CV only.",
        "headline_run_id": head_manifest["run_id"],
        "headline_metrics_sha256": head_manifest["artifact_sha256"]["metrics.json"],
        "followup_config_hash": fcfg.hash(),
        "valuations_sha256": val_digest,
        "price_level": {
            **pl_cfg,
            "definition": "log of the median positive fee of moves where the buying or selling "
            "club played in the PL in the move's cycle season, over [D - window_days, D)",
            "by_cycle_median_level_eur": {
                str(c): float(np.exp(g["league_price_level"].median()))
                for c, g in df.groupby("transfer_cycle")
            },
        },
        "variants": per_variant,
        "variant_order": [v["name"] for v in vs],
        "selected_variant": sel,
        "selected_model": per_variant[sel]["model"],
        "cv_folds": [f.validation_cycle for f in folds],
        "followup_vs_headline": boot,
        "selected_coefficients": coefs.to_dict(orient="records"),
        "window_diagnostic": window,
        "market_value_comparator": market_value,
    }
    write_json(result, fcfg.output_dir / "followup.json")
    return result
