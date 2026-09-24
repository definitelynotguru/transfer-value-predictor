import numpy as np
import pandas as pd
import pytest

from transfer_value.evaluate import (
    baseline_predictions,
    metrics,
    paired_cluster_bootstrap,
    retransform,
    worst_misses,
)
from transfer_value.model import rank_key, select


def test_metrics_by_hand():
    y = np.array([10.0, 20.0, 40.0])
    p = np.array([12.0, 20.0, 30.0])
    m = metrics(y, p, np.log1p(p))
    assert m["mae_eur"] == pytest.approx(4.0)
    assert m["median_ae_eur"] == pytest.approx(2.0)
    assert m["rmse_eur"] == pytest.approx(np.sqrt((4 + 0 + 100) / 3))


def test_baselines_use_train_only_with_position_fallback():
    train = pd.DataFrame({"fee_eur": [1.0, 3.0, 10.0], "position": ["DF", "DF", "FW"]})
    test = pd.DataFrame({"position": ["DF", "GK"]})
    preds, info = baseline_predictions(train, test)
    assert preds["train_median"].tolist() == [3.0, 3.0]
    assert preds["train_mean"][0] == pytest.approx(14 / 3)
    assert preds["train_median_by_position"].tolist() == [2.0, 3.0]
    assert info["position_fallback_rows"] == 1


def test_retransform_clamps_and_counts():
    eur, n = retransform(np.array([-0.5, 0.0, np.log1p(99.0)]))
    assert eur.tolist() == pytest.approx([0.0, 0.0, 99.0])
    assert n == 1
    with pytest.raises(ValueError):
        retransform(np.array([np.nan]))


def test_bootstrap_preserves_cluster_multiplicity():
    # Player a has two transfers; drawing a twice must count four rows, not two.
    clusters = pd.Series(["a", "a", "b"])
    err = {"m": np.array([0.0, 0.0, 10.0]), "b": np.array([5.0, 5.0, 5.0])}
    out = paired_cluster_bootstrap(clusters, err, "m", 2000, 0, 0.95)
    c = out["comparisons"]["b"]
    assert out["resampling_unit"] == "player"
    assert c["observed_delta_mae_eur"] == pytest.approx(10 / 3 - 5)
    # Possible replicate deltas: all-a (-5), all-b (+5), mixed (a,b): (10/3 - 5).
    assert c["ci_low"] == pytest.approx(-5) and c["ci_high"] == pytest.approx(5)


def test_bootstrap_is_seeded():
    clusters = pd.Series(list("abcdefgh"))
    rng = np.random.default_rng(1)
    err = {"m": rng.random(8), "b": rng.random(8)}
    a = paired_cluster_bootstrap(clusters, err, "m", 500, 7, 0.9)
    b = paired_cluster_bootstrap(clusters, err, "m", 500, 7, 0.9)
    assert a == b


def test_worst_misses_are_deterministic():
    df = pd.DataFrame({"transfer_id": ["b", "a", "c"], "abs_error_eur": [5.0, 5.0, 1.0]})
    assert worst_misses(df, 2)["transfer_id"].tolist() == ["a", "b"]


def test_selection_ties_break_by_grid_order():
    rows = [
        {
            "order": 0,
            "name": "linear",
            "family": "linear",
            "mean_log_mae": 0.80970001,
            "std_log_mae": 0.05,
            "eligible": True,
        },
        {
            "order": 1,
            "name": "ridge",
            "family": "ridge",
            "mean_log_mae": 0.80970000,
            "std_log_mae": 0.05,
            "eligible": True,
        },
        {
            "order": 2,
            "name": "en",
            "family": "elastic_net",
            "mean_log_mae": 0.9,
            "std_log_mae": 0.05,
            "eligible": True,
        },
    ]
    finalists, selected = select(rows)
    assert selected["name"] == "linear"
    assert rank_key(rows[0]) < rank_key(rows[1])


def test_ineligible_fits_are_not_selected():
    rows = [
        {
            "order": 0,
            "name": "linear",
            "family": "linear",
            "mean_log_mae": 0.1,
            "std_log_mae": 0.0,
            "eligible": False,
        },
        {
            "order": 1,
            "name": "linear2",
            "family": "linear",
            "mean_log_mae": 0.5,
            "std_log_mae": 0.0,
            "eligible": True,
        },
        {
            "order": 2,
            "name": "ridge",
            "family": "ridge",
            "mean_log_mae": 0.4,
            "std_log_mae": 0.0,
            "eligible": True,
        },
        {
            "order": 3,
            "name": "en",
            "family": "elastic_net",
            "mean_log_mae": 0.6,
            "std_log_mae": 0.0,
            "eligible": True,
        },
    ]
    finalists, selected = select(rows)
    assert finalists["linear"]["name"] == "linear2"
    assert selected["name"] == "ridge"
