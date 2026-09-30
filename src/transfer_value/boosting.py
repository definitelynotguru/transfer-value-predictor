"""Gradient boosting on the headline inputs, with and without monotonic constraints.

A post-holdout check of whether a more flexible model reduces the compression of elite fees.
Selection mirrors the linear grid: training-window CV log-MAE, then fold SD, then grid order.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from transfer_value.features import assert_allowed_inputs
from transfer_value.model import rank_key
from transfer_value.split import Fold

# Appearances is unconstrained: holding minutes fixed, more appearances means more
# substitute outings, and the linear fit gives it a negative sign.
MONOTONE = {
    "goals": 1,
    "assists": 1,
    "minutes": 1,
    "appearances": 0,
    "goals_per90": 1,
    "assists_per90": 1,
    "age": -1,
}
LEARNING_RATE = 0.05
L2 = 1.0


def grid() -> list[dict]:
    """Fixed order; also the tie-breaker."""
    return [
        {"max_depth": d, "min_samples_leaf": leaf, "max_iter": it}
        for d, leaf, it in itertools.product((2, 3), (10, 20), (150, 300))
    ]


def name(params: dict, monotone: bool) -> str:
    kind = "gbm_monotone" if monotone else "gbm"
    return kind + "(" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"


def make_gbm(
    params: dict, numeric: list[str], categorical: list[str], monotone: bool, random_state: int
) -> Pipeline:
    assert_allowed_inputs(numeric + categorical)
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    cst = {c: MONOTONE.get(c, 0) for c in numeric} if monotone else None
    model = HistGradientBoostingRegressor(
        learning_rate=LEARNING_RATE,
        l2_regularization=L2,
        monotonic_cst=cst,
        early_stopping=False,
        random_state=random_state,
        **params,
    )
    return Pipeline([("pre", pre), ("model", model)])


def cross_validate_gbm(
    train: pd.DataFrame,
    folds: list[Fold],
    numeric: list[str],
    categorical: list[str],
    monotone: bool,
    random_state: int,
) -> tuple[dict, list[dict]]:
    X = train[numeric + categorical]
    y = train["fee_log1p"].to_numpy()
    results = []
    for order, params in enumerate(grid()):
        fold_mae = []
        for f in folds:
            pipe = make_gbm(params, numeric, categorical, monotone, random_state)
            pipe.fit(X.loc[f.train_idx], y[f.train_idx])
            fold_mae.append(float(np.mean(np.abs(pipe.predict(X.loc[f.val_idx]) - y[f.val_idx]))))
        results.append(
            {
                "order": order,
                "name": name(params, monotone),
                "params": params,
                "fold_log_mae": fold_mae,
                "mean_log_mae": float(np.mean(fold_mae)),
                "std_log_mae": float(np.std(fold_mae)),
            }
        )
    return min(results, key=rank_key), results


def compression(y_eur: np.ndarray, pred_log: np.ndarray) -> dict:
    """How far predictions reach toward the top of the market on the test set."""
    y_log = np.log1p(y_eur)
    top = y_log >= np.quantile(y_log, 0.9)
    i = int(np.argmax(y_eur))
    return {
        "max_prediction_eur": float(np.expm1(pred_log.max())),
        "largest_fee_eur": float(y_eur[i]),
        "prediction_for_largest_fee_eur": float(np.expm1(pred_log[i])),
        "top_decile_rows": int(top.sum()),
        "top_decile_mean_log_residual": float(np.mean(y_log[top] - pred_log[top])),
        "calibration_slope": (
            float(np.polyfit(pred_log, y_log, 1)[0]) if np.ptp(pred_log) > 0 else None
        ),
    }
