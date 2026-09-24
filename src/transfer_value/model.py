"""Shared preprocessing pipeline, hyperparameter grid, and chronological CV selection."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from transfer_value.features import assert_allowed_inputs
from transfer_value.split import Fold

FAMILIES = ("linear", "ridge", "elastic_net")


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        if not self.params:
            return self.family
        return self.family + "(" + ", ".join(f"{k}={v}" for k, v in self.params.items()) + ")"


def grid(random_state: int) -> list[Candidate]:
    """Fixed order: this order is also the final tie-breaker."""
    out = [Candidate("linear")]
    out += [Candidate("ridge", {"alpha": a}) for a in (0.1, 1.0, 10.0, 100.0)]
    out += [
        Candidate("elastic_net", {"alpha": a, "l1_ratio": r})
        for a in (0.1, 1.0, 10.0)
        for r in (0.15, 0.5, 0.85)
    ]
    return out


def make_estimator(c: Candidate, random_state: int):
    if c.family == "linear":
        return LinearRegression()
    if c.family == "ridge":
        return Ridge(alpha=c.params["alpha"], random_state=random_state)
    if c.family == "elastic_net":
        return ElasticNet(
            alpha=c.params["alpha"],
            l1_ratio=c.params["l1_ratio"],
            max_iter=100_000,
            tol=1e-6,
            selection="cyclic",
            random_state=random_state,
        )
    raise ValueError(f"unknown family {c.family}")


def make_pipeline(
    c: Candidate, numeric: list[str], categorical: list[str], random_state: int
) -> Pipeline:
    assert_allowed_inputs(numeric + categorical)
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("pre", pre), ("model", make_estimator(c, random_state))])


def fit_checked(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> tuple[Pipeline, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X, y)
    issues = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return pipe, issues


def cross_validate(
    train: pd.DataFrame,
    folds: list[Fold],
    numeric: list[str],
    categorical: list[str],
    random_state: int,
) -> list[dict]:
    X = train[numeric + categorical]
    y = train["fee_log1p"].to_numpy()
    results = []
    for order, c in enumerate(grid(random_state)):
        fold_mae, issues = [], []
        for f in folds:
            pipe = make_pipeline(c, numeric, categorical, random_state)
            pipe, warn = fit_checked(clone(pipe), X.loc[f.train_idx], y[f.train_idx])
            pred = pipe.predict(X.loc[f.val_idx])
            if not np.isfinite(pred).all():
                warn.append("non-finite validation predictions")
            issues += warn
            fold_mae.append(float(np.mean(np.abs(pred - y[f.val_idx]))))
        results.append(
            {
                "order": order,
                "name": c.name,
                "family": c.family,
                "params": c.params,
                "fold_log_mae": fold_mae,
                "mean_log_mae": float(np.mean(fold_mae)),
                "std_log_mae": float(np.std(fold_mae)),
                "fit_issues": issues,
                "eligible": not issues,
            }
        )
    return results


TIE_DECIMALS = 4


def rank_key(r: dict) -> tuple:
    # Rounded so float noise (e.g. Ridge alpha=0.1 vs OLS differing by 1e-8) counts as a tie
    # and the fixed grid order decides instead.
    return (
        round(r["mean_log_mae"], TIE_DECIMALS),
        round(r["std_log_mae"], TIE_DECIMALS),
        r["order"],
    )


def select(results: list[dict]) -> tuple[dict[str, dict], dict]:
    """Best eligible candidate per family, and the overall CV-selected candidate."""
    eligible = [r for r in results if r["eligible"]]
    finalists = {}
    for fam in FAMILIES:
        fam_rows = [r for r in eligible if r["family"] == fam]
        if not fam_rows:
            raise RuntimeError(f"no eligible {fam} candidate (all fits had issues)")
        finalists[fam] = min(fam_rows, key=rank_key)
    selected = min(finalists.values(), key=rank_key)
    return finalists, selected


def run_train(cfg) -> dict:
    from transfer_value import __version__
    from transfer_value.features import model_inputs
    from transfer_value.io import (
        frame_fingerprint,
        read_json,
        read_parquet,
        write_joblib,
        write_json,
        write_text,
    )
    from transfer_value.split import chronological_folds, holdout

    cfg.require_pinned()
    fmeta = read_json(cfg.processed_dir / "features_manifest.json")
    feats = read_parquet(cfg.processed_dir / "features.parquet")
    if fmeta["config_hash"] != cfg.hash():
        raise RuntimeError("features were built with a different config; rerun build-features")
    if frame_fingerprint(feats) != fmeta["features_fingerprint"]:
        raise RuntimeError("features.parquet does not match its manifest; rerun build-features")

    rs = int(cfg.runtime["random_state"])
    numeric, categorical = model_inputs(bool(cfg.study["include_cards"]))
    train, _ = holdout(
        feats, pd.Timestamp(cfg.test_start), cfg.split["min_total_rows"], cfg.split["min_test_rows"]
    )
    folds = chronological_folds(
        train, cfg.cv["n_folds"], cfg.cv["min_train_rows"], cfg.cv["min_validation_rows"]
    )
    results = cross_validate(train, folds, numeric, categorical, rs)
    finalists, selected = select(results)

    art = cfg.artifact_dir
    X, y = train[numeric + categorical], train["fee_log1p"].to_numpy()
    coef_frames, bundles = [], {}
    for fam, r in finalists.items():
        c = Candidate(r["family"], r["params"])
        pipe, issues = fit_checked(make_pipeline(c, numeric, categorical, rs), X, y)
        if issues:
            raise RuntimeError(f"final fit of {c.name} had issues: {issues}")
        is_sel = r is selected
        bundle = {
            "pipeline": pipe,
            "family": fam,
            "name": c.name,
            "params": c.params,
            "selected_by_cv": is_sel,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "categories": [
                list(x) for x in pipe.named_steps["pre"].named_transformers_["cat"].categories_
            ],
            "target": "log1p(fee_eur)",
            "retransform": "max(expm1(prediction), 0)",
            "train_rows": len(train),
            "train_date_range": [
                train["transfer_date"].min().date().isoformat(),
                train["transfer_date"].max().date().isoformat(),
            ],
            "features_fingerprint": fmeta["features_fingerprint"],
            "run_id": fmeta["run_id"],
            "source_hashes": fmeta["source_hashes"],
            "config_hash": cfg.hash(),
            "package_version": __version__,
            "random_state": rs,
        }
        bundles[fam] = bundle
        write_joblib(bundle, art / "models" / f"{fam}.joblib")
        coef_frames.append(coefficient_table(pipe, c.name, is_sel))
    write_joblib(bundles[selected["family"]], art / "model.joblib")

    coefs = pd.concat(coef_frames, ignore_index=True)
    write_text(coefs.to_csv(index=False), art / "coefficients.csv")
    cv = {
        "run_id": fmeta["run_id"],
        "objective": "mean fold MAE on log1p(fee_eur); equal fold weights",
        "tie_break": [
            f"mean_log_mae rounded to {TIE_DECIMALS} dp",
            f"std_log_mae rounded to {TIE_DECIMALS} dp",
            "grid order",
        ],
        "folds": [
            {
                "index": f.index,
                "validation_cycle": f.validation_cycle,
                "train_rows": len(f.train_idx),
                "validation_rows": len(f.val_idx),
                "train_date_max": train.loc[f.train_idx, "transfer_date"].max().date().isoformat(),
                "validation_date_range": [
                    train.loc[f.val_idx, "transfer_date"].min().date().isoformat(),
                    train.loc[f.val_idx, "transfer_date"].max().date().isoformat(),
                ],
                "validation_transfer_ids": train.loc[f.val_idx, "transfer_id"].tolist(),
            }
            for f in folds
        ],
        "candidates": results,
        "finalists": {k: v["name"] for k, v in finalists.items()},
        "selected": selected["name"],
        "selected_family": selected["family"],
    }
    write_json(cv, art / "cv_results.json")
    return cv


def coefficient_table(pipe: Pipeline, model_name: str, selected: bool) -> pd.DataFrame:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    est = pipe.named_steps["model"]
    names = list(pre.get_feature_names_out())
    scaler: StandardScaler = pre.named_transformers_["num"]
    num_cols = list(scaler.feature_names_in_)
    rows = []
    for name, coef in zip(names, np.ravel(est.coef_), strict=True):
        kind, raw = name.split("__", 1)
        row = {
            "model": model_name,
            "selected": selected,
            "term": raw,
            "kind": kind,
            "coef_log1p": float(coef),
            "multiplier_on_1p_fee": float(np.exp(coef)),
        }
        if kind == "num":
            i = num_cols.index(raw)
            row.update(
                train_mean=float(scaler.mean_[i]),
                train_sd=float(scaler.scale_[i]),
                constant=bool(scaler.var_[i] == 0),
            )
        rows.append(row)
    rows.append(
        {
            "model": model_name,
            "selected": selected,
            "term": "(intercept)",
            "kind": "intercept",
            "coef_log1p": float(est.intercept_),
            "multiplier_on_1p_fee": float(np.exp(est.intercept_)),
        }
    )
    return pd.DataFrame(rows)
