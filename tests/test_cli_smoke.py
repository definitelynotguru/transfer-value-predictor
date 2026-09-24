"""End-to-end CLI on the offline fixture snapshot. No network."""

import json

import joblib
import pytest
from typer.testing import CliRunner

from conftest import make_env
from transfer_value.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    return root, str(make_env(root))


def _ok(args):
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output + str(r.exception)
    return r.output


def test_help():
    assert "pipeline" in _ok(["--help"])


def test_missing_config_fails():
    r = runner.invoke(app, ["ingest", "--config", "does-not-exist.yaml"])
    assert r.exit_code == 1


def test_train_before_features_fails(env):
    root, cfg = env
    r = runner.invoke(app, ["train", "--config", cfg])
    assert r.exit_code == 1


def test_pipeline_and_predict(env):
    root, cfg = env
    out = _ok(["pipeline", "--config", cfg])
    assert "pipeline ok" in out
    art = root / "artifacts"
    m = json.loads((art / "metrics.json").read_text())
    assert set(m["methods"]) == {
        "train_median",
        "train_mean",
        "train_median_by_position",
        "linear",
        "ridge",
        "elastic_net",
    }
    assert sum(v["selected_by_cv"] for v in m["methods"].values()) == 1
    rows = {v["rows"] for v in m["methods"].values()}
    assert len(rows) == 1  # every method scored on the same test rows
    manifest = json.loads((art / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["run_id"] == m["run_id"]
    funnel = json.loads((root / "data/processed/funnel.json").read_text())
    assert funnel["train_rows"] + funnel["test_rows"] == funnel["final_rows"]
    for f in ("predicted_vs_actual", "residuals_vs_predicted", "residual_distribution"):
        assert (art / "figures" / f"{f}.png").exists()

    out = _ok(["predict", "--player", "alex example", "--config", cfg])
    assert "Alex Example" in out and "hypothetical" in out.lower()


def test_rerun_is_deterministic(env):
    root, cfg = env
    art = root / "artifacts"
    before = json.loads((art / "metrics.json").read_text())
    _ok(["build-features", "--config", cfg])
    _ok(["train", "--config", cfg])
    _ok(["evaluate", "--config", cfg])
    after = json.loads((art / "metrics.json").read_text())
    assert before == after


def test_predict_refuses_bad_inputs(env):
    root, cfg = env
    r = runner.invoke(app, ["predict", "--player", "Nobody", "--config", cfg])
    assert r.exit_code == 1 and "no player" in r.output
    r = runner.invoke(
        app, ["predict", "--player", "Alex Example", "--date", "2021-01-01", "--config", cfg]
    )
    assert r.exit_code == 1 and "future" in r.output
    r = runner.invoke(
        app, ["predict", "--player", "Alex Example", "--date", "2030-01-01", "--config", cfg]
    )
    assert r.exit_code == 1 and "coverage" in r.output


def test_stale_model_is_rejected(env):
    root, cfg = env
    bundle = joblib.load(root / "artifacts/models/linear.joblib")
    bundle["features_fingerprint"] = "stale"
    joblib.dump(bundle, root / "artifacts/models/linear.joblib")
    r = runner.invoke(app, ["evaluate", "--config", cfg])
    assert r.exit_code == 1 and "stale" in r.output
