"""Holdout evaluation: baselines, metrics, paired bootstrap, worst misses, figures, manifest."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from transfer_value import plots
from transfer_value.io import (
    code_revision,
    environment_info,
    frame_fingerprint,
    read_json,
    read_parquet,
    sha256_file,
    write_json,
    write_parquet,
    write_text,
)
from transfer_value.model import FAMILIES
from transfer_value.split import holdout, split_summary

BASELINES = ("train_median", "train_mean", "train_median_by_position")
LABELS = {
    "train_median": "Train median fee",
    "train_mean": "Train mean fee",
    "train_median_by_position": "Train median fee by position",
    "linear": "LinearRegression",
    "ridge": "Ridge",
    "elastic_net": "ElasticNet",
}


def retransform(pred_log: np.ndarray) -> tuple[np.ndarray, int]:
    if not np.isfinite(pred_log).all():
        raise ValueError("non-finite log predictions")
    eur = np.expm1(pred_log)
    if not np.isfinite(eur).all():
        raise ValueError("expm1 overflow in predictions")
    return np.maximum(eur, 0.0), int((eur < 0).sum())


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict]:
    med = float(train["fee_eur"].median())
    mean = float(train["fee_eur"].mean())
    by_pos = train.groupby("position")["fee_eur"].median()
    pos_pred = test["position"].map(by_pos)
    fallback = int(pos_pred.isna().sum())
    preds = {
        "train_median": np.full(len(test), med),
        "train_mean": np.full(len(test), mean),
        "train_median_by_position": pos_pred.fillna(med).to_numpy(dtype=float),
    }
    info = {
        "train_median_eur": med,
        "train_mean_eur": mean,
        "train_median_by_position_eur": {k: float(v) for k, v in by_pos.items()},
        "position_fallback_rows": fallback,
    }
    return preds, info


def metrics(y: np.ndarray, pred_eur: np.ndarray, pred_log: np.ndarray) -> dict:
    err = y - pred_eur
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae_eur": float(np.mean(np.abs(err))),
        "median_ae_eur": float(np.median(np.abs(err))),
        "rmse_eur": float(np.sqrt(np.mean(err**2))),
        "log_mae": float(np.mean(np.abs(np.log1p(y) - pred_log))),
        "r2_eur": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else None,
        "rows": len(y),
    }


def paired_cluster_bootstrap(
    clusters: pd.Series,
    abs_err: dict[str, np.ndarray],
    reference: str,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict:
    """CI on MAE(reference) − MAE(other) with identical resampled clusters per replicate.

    Units are players: each draw samples unique test players with replacement and keeps all
    their transfers, repeated as often as the player is drawn. With no repeated players this is
    the ordinary paired row bootstrap. Models stay fixed; this is holdout sampling noise only.
    """
    codes, uniq = pd.factorize(clusters, sort=True)
    n_c = len(uniq)
    counts = np.bincount(codes, minlength=n_c)
    sums = {k: np.bincount(codes, weights=v, minlength=n_c) for k, v in abs_err.items()}
    draws = np.random.default_rng(seed).integers(0, n_c, size=(replicates, n_c))
    n_rows = counts[draws].sum(axis=1)
    ref_mae = sums[reference][draws].sum(axis=1) / n_rows
    alpha = (1 - confidence) / 2
    out = {}
    for k in abs_err:
        if k == reference:
            continue
        delta = ref_mae - sums[k][draws].sum(axis=1) / n_rows
        observed = float(abs_err[reference].mean() - abs_err[k].mean())
        lo, hi = (float(x) for x in np.quantile(delta, [alpha, 1 - alpha]))
        if hi < 0:
            verdict = "model MAE lower; interval excludes zero"
        elif lo > 0:
            verdict = "baseline MAE lower; interval excludes zero"
        else:
            verdict = "too uncertain to call; interval includes zero"
        out[k] = {
            "observed_delta_mae_eur": observed,
            "ci_low": lo,
            "ci_high": hi,
            "verdict": verdict,
        }
    return {
        "reference": reference,
        "delta_definition": "MAE(reference model) - MAE(comparison); negative favors the model",
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "resampling_unit": "player" if counts.max() > 1 else "transfer (no repeated players)",
        "unique_test_players": n_c,
        "comparisons": out,
    }


def worst_misses(pred: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return pred.sort_values(["abs_error_eur", "transfer_id"], ascending=[False, True]).head(k)


def worst_misses_markdown(rows: pd.DataFrame, model: str) -> str:
    lines = [
        f"# Five largest absolute errors ({model})",
        "",
        "Residual = reported − predicted. Positive means the model under-predicted.",
        "",
    ]
    for i, r in enumerate(rows.itertuples(), 1):
        lines += [
            f"## {i}. {r.name} ({r.from_club_name} → {r.to_club_name}, "
            f"{r.transfer_date.date().isoformat()})",
            "",
            f"- Reported fee €{r.fee_eur / 1e6:.1f}m; predicted €{r.prediction_eur / 1e6:.1f}m; "
            f"residual {'+' if r.residual_eur >= 0 else '−'}€{abs(r.residual_eur) / 1e6:.1f}m",
            f"- Age {int(r.age)}, position {r.position} ({r.position_source}), lookback seasons "
            f"{r.lookback_season_ids}",
            f"- Lookback: {int(r.minutes)} min, {int(r.appearances)} apps, {int(r.goals)} goals, "
            f"{int(r.assists)} assists ({r.goals_per90:.2f} G/90, {r.assists_per90:.2f} A/90)",
            f"- IDs: player {r.player_id}, transfer {r.transfer_id}",
            "",
        ]
    return "\n".join(lines)


def run_evaluate(cfg) -> dict:
    cfg.require_pinned()
    art = cfg.artifact_dir
    fmeta = read_json(cfg.processed_dir / "features_manifest.json")
    feats = read_parquet(cfg.processed_dir / "features.parquet")
    fp = frame_fingerprint(feats)
    cv = read_json(art / "cv_results.json")
    bundles = {}
    for fam in FAMILIES:
        path = art / "models" / f"{fam}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"missing artifact: {path}; run train")
        b = joblib.load(path)
        if b["features_fingerprint"] != fp or b["config_hash"] != cfg.hash():
            raise RuntimeError(f"{fam} model is stale relative to features/config; rerun train")
        bundles[fam] = b
    if cv["run_id"] != fmeta["run_id"] or fmeta["features_fingerprint"] != fp:
        raise RuntimeError("cv_results/features mismatch; rerun build-features and train")
    selected = cv["selected_family"]

    t = pd.Timestamp(cfg.test_start)
    train, test = holdout(feats, t, cfg.split["min_total_rows"], cfg.split["min_test_rows"])
    y = test["fee_eur"].to_numpy(dtype=float)

    preds, base_info = baseline_predictions(train, test)
    methods, abs_err, clamped = {}, {}, {}
    for name in BASELINES:
        methods[name] = metrics(y, preds[name], np.log1p(preds[name]))
        abs_err[name] = np.abs(y - preds[name])
    raw_logs = {}
    for fam, b in bundles.items():
        cols = b["numeric_features"] + b["categorical_features"]
        raw_logs[fam] = b["pipeline"].predict(test[cols])
        eur, n_clamped = retransform(raw_logs[fam])
        preds[fam] = eur
        clamped[fam] = n_clamped
        methods[fam] = metrics(y, eur, raw_logs[fam])
        abs_err[fam] = np.abs(y - eur)
    for name in methods:
        methods[name]["label"] = LABELS[name]
        methods[name]["selected_by_cv"] = name == selected
        methods[name]["clamped_at_zero"] = clamped.get(name, 0)
        if name in bundles:
            methods[name]["params"] = bundles[name]["params"]

    rt = cfg.runtime
    boot = paired_cluster_bootstrap(
        test["player_id"],
        {k: abs_err[k] for k in (selected, *BASELINES)},
        selected,
        int(rt["bootstrap_replicates"]),
        int(rt["bootstrap_seed"]),
        float(rt["bootstrap_confidence"]),
    )

    keep = [
        "transfer_id",
        "player_id",
        "name",
        "transfer_date",
        "transfer_cycle",
        "from_club_name",
        "to_club_name",
        "fee_eur",
        "position",
        "position_source",
        "position_is_proxy",
        "age",
        "goals",
        "assists",
        "minutes",
        "appearances",
        "goals_per90",
        "assists_per90",
        "lookback_season_ids",
        "lookback_minutes",
    ]
    out = test[keep].copy()
    out["fee_eur"] = out["fee_eur"].astype(float)
    for name in methods:
        out[f"pred_eur__{name}"] = preds[name]
    for fam in bundles:
        out[f"pred_log__{fam}"] = raw_logs[fam]
    out["prediction_log"] = raw_logs[selected]
    out["prediction_eur"] = preds[selected]
    out["residual_eur"] = out["fee_eur"] - out["prediction_eur"]
    out["abs_error_eur"] = out["residual_eur"].abs()
    write_parquet(out, art / "test_predictions.parquet")

    misses = worst_misses(out)
    model_label = f"{LABELS[selected]} {bundles[selected]['params'] or ''}".strip()
    miss_cols = keep + ["prediction_eur", "residual_eur", "abs_error_eur"]
    write_json(misses[miss_cols].to_dict(orient="records"), art / "worst_misses.json")
    write_text(worst_misses_markdown(misses, model_label), art / "worst_misses.md")

    figs = art / "figures"
    plots.predicted_vs_actual(out, model_label, figs / "predicted_vs_actual.png")
    plots.residuals_vs_predicted(out, model_label, figs / "residuals_vs_predicted.png")
    plots.residual_distribution(out, model_label, figs / "residual_distribution.png")

    total_proxy = float(feats["position_is_proxy"].mean())
    result = {
        "run_id": fmeta["run_id"],
        "question": "How well do recent PL performance, age and position explain PL-active "
        "players' reported transfer fees on a later time holdout?",
        "headline_metric": "test MAE in EUR after expm1 (selection used CV log-MAE)",
        "study_window": fmeta["study_window"],
        "split": split_summary(train, test, t),
        "position_proxy_pct_total": round(100 * total_proxy, 2),
        "selected_model": {
            "family": selected,
            "name": bundles[selected]["name"],
            "params": bundles[selected]["params"],
            "selection": cv["objective"],
        },
        "cv_finalists": {
            fam: next(c for c in cv["candidates"] if c["name"] == name)
            for fam, name in cv["finalists"].items()
        },
        "methods": methods,
        "baselines": base_info,
        "bootstrap": boot,
        "retransformation": "prediction_eur = max(expm1(prediction_log), 0); no smearing "
        "correction, so this is not an expected-fee estimator",
    }
    write_json(result, art / "metrics.json")

    produced = [
        "metrics.json",
        "test_predictions.parquet",
        "worst_misses.json",
        "worst_misses.md",
        "cv_results.json",
        "coefficients.csv",
        "model.joblib",
        "figures/predicted_vs_actual.png",
        "figures/residuals_vs_predicted.png",
        "figures/residual_distribution.png",
    ]
    produced += [f"models/{f}.joblib" for f in FAMILIES]
    manifest = {
        "schema_version": 1,
        "run_id": fmeta["run_id"],
        "status": "complete",
        "code_revision": code_revision(cfg.root),
        "environment": environment_info(cfg.root),
        "config": cfg.raw,
        "config_hash": cfg.hash(),
        "source_hashes": fmeta["source_hashes"],
        "features_fingerprint": fp,
        "seed": int(rt["random_state"]),
        "selected_model": bundles[selected]["name"],
        "target": "log1p(fee_eur)",
        "retransform": "max(expm1(prediction), 0)",
        "split": result["split"],
        "artifact_sha256": {p: sha256_file(art / p) for p in produced},
    }
    write_json(manifest, art / "manifest.json")
    return result
