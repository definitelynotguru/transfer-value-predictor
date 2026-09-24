from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.builder import build  # noqa: E402
from transfer_value.config import RAW_FILES, load_config  # noqa: E402
from transfer_value.dataset import assemble, canonical_tables  # noqa: E402
from transfer_value.source import sha256_file, write_source_manifest  # noqa: E402

FIXTURE_CONFIG = {
    "fixture": True,
    "source": {"base_url": "file://fixture", "files": list(RAW_FILES.values()), "sha256": {}},
    "data": {
        "raw_dir": "data/raw",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "artifact_dir": "artifacts",
    },
    "competition": {"code": "premier-league", "ids": ["GB1"], "games_per_season": 4},
    "fees": {"currency": "EUR", "encoding": "positive_is_paid_permanent"},
    "study": {
        "source_seasons": [2017, 2018, 2019, 2020, 2021, 2022, 2023],
        "season_ids": [2019, 2020, 2021, 2022, 2023],
        "transfer_start": "2019-05-13",
        "transfer_end_exclusive": "2024-05-20",
        "min_minutes": 90,
        "include_cards": False,
        "position_policy": "historical_then_proxy",
        "prediction_as_of": "2024-05-20",
    },
    "split": {"test_start": "2022-05-23", "min_total_rows": 6, "min_test_rows": 2},
    "cv": {"n_folds": 1, "min_train_rows": 2, "min_validation_rows": 1},
    "runtime": {
        "random_state": 42,
        "bootstrap_replicates": 200,
        "bootstrap_seed": 42,
        "bootstrap_confidence": 0.95,
    },
}


VALUATIONS_FILE = "player_valuations.csv.gz"

FIXTURE_FOLLOWUP = {
    "headline_config": "config.yaml",
    "headline_run_id": None,
    "source": {"base_url": "file://fixture", "files": [VALUATIONS_FILE], "sha256": {}},
    "data": {"raw_dir": "data/raw", "output_dir": "artifacts/followup"},
    "price_level": {"window_days": 365, "min_transfers": 1},
    "market_value": {"file": VALUATIONS_FILE, "max_staleness_days": 365},
}


def make_env(root: Path, overrides: dict | None = None) -> Path:
    raw = root / "data" / "raw"
    build(raw)
    entries = [
        {
            "file": f,
            "url": f"file://fixture/{f}",
            "bytes": (raw / f).stat().st_size,
            "sha256": sha256_file(raw / f),
            "acquired_utc": "fixture",
        }
        for f in [*RAW_FILES.values(), VALUATIONS_FILE]
    ]
    write_source_manifest(raw / "SOURCE.md", entries, FIXTURE_CONFIG["source"])
    cfg = {**FIXTURE_CONFIG, **(overrides or {})}
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "followup.yaml").write_text(yaml.safe_dump(FIXTURE_FOLLOWUP, sort_keys=False))
    return path


@pytest.fixture(scope="session")
def fixture_config_path(tmp_path_factory) -> Path:
    return make_env(tmp_path_factory.mktemp("env"))


@pytest.fixture(scope="session")
def cfg(fixture_config_path):
    return load_config(fixture_config_path)


@pytest.fixture(scope="session")
def tables(cfg):
    t, _ = canonical_tables(cfg, cfg.raw_dir)
    return t


@pytest.fixture(scope="session")
def assembled(cfg, tables):
    return assemble(tables, cfg)


@pytest.fixture(scope="session")
def candidates(assembled):
    return assembled["candidates"].set_index("player_id")
