"""Split-conformal prediction intervals calibrated on training-window CV residuals.

Calibration scores are absolute out-of-fold residuals on the log1p scale from the same
expanding chronological folds used for model selection, so the test set never touches the
interval width. The interval is symmetric in log1p(fee) and maps exactly to euros because
expm1 is monotone. Coverage is marginal (over transfers), not per player, and it assumes the
test period errors look like the validation-cycle errors, which drift can break.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from transfer_value.model import Candidate, fit_checked, make_pipeline
from transfer_value.split import Fold


def oof_residuals(
    train: pd.DataFrame,
    folds: list[Fold],
    numeric: list[str],
    categorical: list[str],
    candidate: Candidate,
    random_state: int,
) -> tuple[np.ndarray, list[float]]:
    """Validation residuals log1p(fee) - prediction, concatenated in fold order."""
    X = train[numeric + categorical]
    y = train["fee_log1p"].to_numpy()
    res, fold_mae = [], []
    for f in folds:
        pipe, _ = fit_checked(
            make_pipeline(candidate, numeric, categorical, random_state),
            X.loc[f.train_idx],
            y[f.train_idx],
        )
        r = y[f.val_idx] - pipe.predict(X.loc[f.val_idx])
        res.append(r)
        fold_mae.append(float(np.mean(np.abs(r))))
    return np.concatenate(res), fold_mae


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """The ceil((n + 1) * level)-th smallest score; infinite when n is too small."""
    n = len(scores)
    k = math.ceil((n + 1) * level)
    if k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


def interval_eur(pred_log: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(np.expm1(pred_log - q), 0.0), np.expm1(pred_log + q)


def coverage(y_log: np.ndarray, pred_log: np.ndarray, q: float) -> dict:
    r = y_log - pred_log
    return {
        "rows": len(r),
        "coverage": float(np.mean(np.abs(r) <= q)),
        "share_above_upper": float(np.mean(r > q)),
        "share_below_lower": float(np.mean(r < -q)),
    }


def summarize(
    scores: np.ndarray,
    test: pd.DataFrame,
    pred_log: np.ndarray,
    level: float,
    groups: tuple[str, ...] = ("transfer_cycle", "position", "window"),
) -> dict:
    q = conformal_quantile(scores, level)
    y_log = np.log1p(test["fee_eur"].to_numpy(dtype=float))
    lo, hi = interval_eur(pred_log, q)
    out = {
        "level": level,
        "calibration_rows": len(scores),
        "q_log": q,
        "factor": math.exp(q),
        "test": coverage(y_log, pred_log, q),
        "median_prediction_eur": float(np.median(np.expm1(pred_log))),
        "median_lower_eur": float(np.median(lo)),
        "median_upper_eur": float(np.median(hi)),
    }
    for g in groups:
        if g in test:
            keys = test[g].astype(str).to_numpy()
            out[f"test_by_{g}"] = {
                k: coverage(y_log[keys == k], pred_log[keys == k], q) for k in sorted(set(keys))
            }
    return out
