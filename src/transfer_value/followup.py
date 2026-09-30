"""Post-holdout follow-up: exposure-normalized inputs, time terms, and a market-value comparator.

Everything here was designed after the headline holdout had been scored, so none of it is a
headline result. The headline artifacts are read, verified, and never rewritten. Variant
selection still uses only chronological CV inside the training window; the test set is scored
once per variant afterwards and reported as a labelled follow-up.

Transfermarkt market value enters only as a comparison row. It is never a model input.

A second round, added after the first follow-up had also been scored, adds as-of context
inputs from the raw all-competition tables, split-conformal intervals, and a gradient boosting
check. The first-round variants and their selection are unchanged by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from transfer_value import boosting, conformal
from transfer_value.config import Config, config_hash, load_config
from transfer_value.context import (
    CONTEXT_FEATURES,
    CONTEXT_GROUPS,
    RAW_CONTEXT_FILES,
    context_columns,
    load_raw_context,
)
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
    for key in (
        "headline_config",
        "source",
        "data",
        "price_level",
        "market_value",
        "context",
        "conformal",
    ):
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


def context_variants(first_round: list[dict]) -> list[dict]:
    """Each first-round input set plus the context inputs, in the same order."""
    out = [
        {"name": f"{v['name']}+context", "numeric": v["numeric"] + CONTEXT_FEATURES}
        for v in first_round
    ]
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


def _cv_variants(train, folds, vs, categorical, rs) -> list[dict]:
    rows = []
    for vi, v in enumerate(vs):
        results = cross_validate(train, folds, v["numeric"], categorical, rs)
        best = min((r for r in results if r["eligible"]), key=rank_key)
        rows.append({"variant": v["name"], "variant_order": vi, "best": best, "all": results})
    return rows


def _pick(cv_rows: list[dict]) -> dict:
    def key(row):
        m, s, order = rank_key(row["best"])
        return (m, s, row["variant_order"], order)

    return min(cv_rows, key=key)


def _score_variants(cv_rows, vs, chosen, categorical, train, test, rs) -> dict:
    y = test["fee_eur"].to_numpy(dtype=float)
    out = {"per_variant": {}, "log": {}, "eur": {}, "in_sample": {}, "pipes": {}, "cand": {}}
    for row, v in zip(cv_rows, vs, strict=True):
        c = Candidate(row["best"]["family"], row["best"]["params"])
        pipe, pl = _fit_predict(c, v["numeric"], categorical, train, test, rs)
        eur, clamped = retransform(pl)
        out["log"][v["name"]], out["eur"][v["name"]] = pl, eur
        out["pipes"][v["name"]], out["cand"][v["name"]] = pipe, c
        out["in_sample"][v["name"]] = train["fee_log1p"] - pipe.predict(
            train[v["numeric"] + categorical]
        )
        out["per_variant"][v["name"]] = {
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
    return out


def _context_frame(fcfg: FollowupConfig, cfg: Config, df, tables, source_hashes) -> pd.DataFrame:
    names = list(RAW_CONTEXT_FILES)
    digests = verify_raw_files(cfg.raw_dir, names)
    for n, d in digests.items():
        if source_hashes.get(n) != d:
            raise RuntimeError(f"{n} differs from the bytes behind the headline run")
    raw = load_raw_context(cfg.raw_dir, set(df["player_id"]))
    return context_columns(
        df,
        tables["seasons"],
        tables["appearances"],
        raw,
        cfg.competition_ids,
        list(fcfg.raw["context"]["europe_competitions"]),
    )


def _worst_misses(test: pd.DataFrame, pred_eur: np.ndarray, n: int = 5) -> list[int]:
    """Row positions of the n largest absolute euro errors, ties broken by transfer_id."""
    err = pd.DataFrame(
        {
            "abs": np.abs(test["fee_eur"].to_numpy(dtype=float) - pred_eur),
            "tid": test["transfer_id"].to_numpy(),
            "pos": np.arange(len(test)),
        }
    )
    return err.sort_values(["abs", "tid"], ascending=[False, True])["pos"].head(n).tolist()


def _boosting(train, test, folds, numeric, categorical, head_log, rs) -> dict:
    y = test["fee_eur"].to_numpy(dtype=float)
    out = {
        "note": "Post-holdout check on the headline inputs. Test columns are descriptive only.",
        "learning_rate": boosting.LEARNING_RATE,
        "l2_regularization": boosting.L2,
        "monotone_constraints": boosting.MONOTONE,
        "grid": boosting.grid(),
        "headline_compression": boosting.compression(y, head_log),
    }
    for mono in (False, True):
        best, _ = boosting.cross_validate_gbm(train, folds, numeric, categorical, mono, rs)
        pipe = boosting.make_gbm(best["params"], numeric, categorical, mono, rs)
        pipe.fit(train[numeric + categorical], train["fee_log1p"].to_numpy())
        pl = pipe.predict(test[numeric + categorical])
        eur, _ = retransform(pl)
        out["gbm_monotone" if mono else "gbm"] = {
            "model": best["name"],
            "cv_mean_log_mae": best["mean_log_mae"],
            "cv_fold_log_mae": best["fold_log_mae"],
            "test": metrics(y, eur, pl),
            "compression": boosting.compression(y, pl),
        }
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
    df = _context_frame(fcfg, cfg, df, tables, head_manifest["source_hashes"])

    rs = int(cfg.runtime["random_state"])
    numeric, categorical = model_inputs(bool(cfg.study["include_cards"]))
    train, test = holdout(
        df, pd.Timestamp(cfg.test_start), cfg.split["min_total_rows"], cfg.split["min_test_rows"]
    )
    folds = chronological_folds(
        train, cfg.cv["n_folds"], cfg.cv["min_train_rows"], cfg.cv["min_validation_rows"]
    )

    vs = variants(numeric)
    cv_rows = _cv_variants(train, folds, vs, categorical, rs)
    head_row = cv_rows[0]["best"]
    if head_row["name"] != head_cv["selected"] or not np.isclose(
        head_row["mean_log_mae"],
        next(c["mean_log_mae"] for c in head_cv["candidates"] if c["name"] == head_cv["selected"]),
    ):
        raise AssertionError("headline variant does not reproduce the headline CV selection")
    chosen = _pick(cv_rows)

    y = test["fee_eur"].to_numpy(dtype=float)
    r1 = _score_variants(cv_rows, vs, chosen, categorical, train, test, rs)
    per_variant, preds_log, preds_eur = r1["per_variant"], r1["log"], r1["eur"]
    in_sample, pipes = r1["in_sample"], r1["pipes"]

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

    # Round 2: context inputs. Selection again uses training-window CV only.
    vs2 = context_variants(vs)
    cv_rows2 = _cv_variants(train, folds, vs2, categorical, rs)
    chosen2 = _pick(cv_rows2)
    r2 = _score_variants(cv_rows2, vs2, chosen2, categorical, train, test, rs)
    sel2 = chosen2["variant"]
    sel2_numeric = next(v["numeric"] for v in vs2 if v["name"] == sel2)
    ablation = {}
    for group, cols in CONTEXT_GROUPS.items():
        kept = [c for c in sel2_numeric if c not in cols]
        best = min(
            (r for r in cross_validate(train, folds, kept, categorical, rs) if r["eligible"]),
            key=rank_key,
        )
        ablation[group] = {
            "dropped": cols,
            "model": best["name"],
            "cv_mean_log_mae": best["mean_log_mae"],
        }
    ae["enriched"] = np.abs(y - r2["eur"][sel2])
    boot2 = paired_cluster_bootstrap(
        test["player_id"],
        {k: ae[k] for k in ("enriched", "headline", "followup")},
        "enriched",
        *boot_args,
        names=("enriched", "other"),
    )
    worst = _worst_misses(test, preds_eur["headline"])
    worst_rows = [
        {
            "transfer_id": test["transfer_id"].iat[i],
            "name": test["name"].iat[i],
            "fee_eur": float(y[i]),
            "headline_eur": float(preds_eur["headline"][i]),
            "followup_eur": float(preds_eur[sel][i]),
            "enriched_eur": float(r2["eur"][sel2][i]),
            **{c: float(test[c].iat[i]) for c in CONTEXT_FEATURES},
        }
        for i in worst
    ]
    context_round = {
        "note": "Second post-holdout round, designed after both the headline and the first "
        "follow-up had been scored. Variant selection used training-window CV only; test "
        "columns for non-selected variants are descriptive only.",
        "definitions": {
            "europe_minutes": "minutes in "
            + ", ".join(fcfg.raw["context"]["europe_competitions"])
            + " inside the lookback date window",
            "other_league_minutes": "minutes in first-tier domestic leagues other than the study "
            "league inside the lookback date window",
            "other_league_goals": "goals in those leagues inside the same window",
            "team_points_per_game": "mean league points the player's club took in the lookback "
            "PL matches he played",
            "lookback_date_window": "[first match of the earlier completed lookback season, D)",
        },
        "variants": r2["per_variant"],
        "variant_order": [v["name"] for v in vs2],
        "selected_variant": sel2,
        "selected_model": r2["per_variant"][sel2]["model"],
        "first_round_selected_cv_log_mae": per_variant[sel]["cv_mean_log_mae"],
        "ablation_cv": ablation,
        "vs_headline_and_followup": boot2,
        "share_with_europe_minutes": {
            "train": float((train["europe_minutes"] > 0).mean()),
            "test": float((test["europe_minutes"] > 0).mean()),
        },
        "share_with_other_league_minutes": {
            "train": float((train["other_league_minutes"] > 0).mean()),
            "test": float((test["other_league_minutes"] > 0).mean()),
        },
        "headline_worst_misses": worst_rows,
        "selected_coefficients": coefficient_table(
            r2["pipes"][sel2], r2["per_variant"][sel2]["model"], True
        ).to_dict(orient="records"),
    }

    # Split-conformal intervals from out-of-fold CV residuals.
    conf_cfg = fcfg.raw["conformal"]
    levels = [float(x) for x in conf_cfg["levels"]]
    predict_level = float(conf_cfg["predict_level"])
    if predict_level not in levels:
        raise ValueError("conformal.predict_level must be one of conformal.levels")
    conf_models = {
        "headline": ("headline", r1, vs[0]["numeric"]),
        "followup": (sel, r1, next(v["numeric"] for v in vs if v["name"] == sel)),
        "enriched": (sel2, r2, sel2_numeric),
    }
    conf_out, bounds = {}, {}
    for key, (vname, rr, cols) in conf_models.items():
        pv = rr["per_variant"][vname]
        scores, fold_mae = conformal.oof_residuals(
            train, folds, cols, categorical, rr["cand"][vname], rs
        )
        if not np.allclose(fold_mae, pv["cv_fold_log_mae"]):
            raise AssertionError(f"{key}: out-of-fold residuals do not reproduce CV")
        conf_out[key] = {
            "variant": vname,
            "model": pv["model"],
            "calibration_mean_residual": float(np.mean(scores)),
            "levels": {
                f"{lv:.2f}": conformal.summarize(np.abs(scores), test, rr["log"][vname], lv)
                for lv in levels
            },
        }
        for lv in levels:
            q = conf_out[key]["levels"][f"{lv:.2f}"]["q_log"]
            bounds[(key, lv)] = conformal.interval_eur(rr["log"][vname], q)
    hq = conf_out["headline"]["levels"][f"{predict_level:.2f}"]["q_log"]
    lo, hi = conformal.interval_eur(preds_log["headline"], hq)
    conf_out["headline_worst_misses"] = [
        {
            "name": test["name"].iat[i],
            "fee_eur": float(y[i]),
            "lower_eur": float(lo[i]),
            "upper_eur": float(hi[i]),
            "covered": bool(lo[i] <= y[i] <= hi[i]),
        }
        for i in worst
    ]
    conf_out["predict_level"] = predict_level
    conf_out["method"] = (
        "split conformal on |log1p(fee) - prediction| from the expanding CV folds' validation "
        "cycles; interval = expm1(prediction -/+ q), lower bound clamped at zero"
    )

    boost = _boosting(train, test, folds, numeric, categorical, preds_log["headline"], rs)

    idx_test = test.index
    log_res_test = {
        "headline": pd.Series(np.log1p(y) - preds_log["headline"], index=idx_test),
        "followup": pd.Series(np.log1p(y) - preds_log[sel], index=idx_test),
    }
    abs_err_test = {k: pd.Series(ae[k], index=idx_test) for k in ("followup", "headline")}
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
        "enriched": r2["eur"][sel2][matched],
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
        for ref in ("headline", "followup", "enriched")
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
    out_pred[f"pred_eur__{sel2}"] = r2["eur"][sel2]
    for (key, lv), (b_lo, b_hi) in bounds.items():
        out_pred[f"lower_eur__{key}__{lv:.2f}"] = b_lo
        out_pred[f"upper_eur__{key}__{lv:.2f}"] = b_hi
    out_pred["market_value_eur"] = mv["market_value_eur"].to_numpy()
    out_pred["valuation_age_days"] = mv["valuation_age_days"].to_numpy()
    fcfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(out_pred, fcfg.output_dir / "test_predictions.parquet")

    hl = conf_out["headline"]["levels"][f"{predict_level:.2f}"]
    write_json(
        {
            "note": "Training-set conformal interval for tvp predict. Marginal coverage over "
            "transfers, not a valuation range for any one player.",
            "headline_run_id": head_manifest["run_id"],
            "features_fingerprint": head_manifest["features_fingerprint"],
            "model": per_variant["headline"]["model"],
            "level": predict_level,
            "q_log": hl["q_log"],
            "factor": hl["factor"],
            "calibration_rows": hl["calibration_rows"],
            "followup_config_hash": fcfg.hash(),
        },
        fcfg.output_dir / "headline_interval.json",
    )

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
        "context_round": context_round,
        "conformal": conf_out,
        "boosting": boost,
    }
    write_json(result, fcfg.output_dir / "followup.json")
    return result
