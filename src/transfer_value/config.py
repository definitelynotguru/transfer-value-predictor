"""Config loading, validation, and raw schema constants (the single source of truth)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Raw columns each table must provide. Ingest fails loud if any are missing.
SCHEMA: dict[str, list[str]] = {
    "transfers": [
        "player_id",
        "transfer_date",
        "transfer_season",
        "from_club_id",
        "to_club_id",
        "from_club_name",
        "to_club_name",
        "transfer_fee",
    ],
    "appearances": [
        "appearance_id",
        "game_id",
        "player_id",
        "player_club_id",
        "date",
        "competition_id",
        "goals",
        "assists",
        "minutes_played",
        "yellow_cards",
        "red_cards",
    ],
    "players": ["player_id", "name", "date_of_birth", "position"],
    "games": ["game_id", "competition_id", "season", "date", "home_club_goals", "away_club_goals"],
    "competitions": ["competition_id", "competition_code", "type"],
    "game_lineups": ["game_id", "player_id", "position"],
}

RAW_FILES: dict[str, str] = {name: f"{name}.csv.gz" for name in SCHEMA}

POSITIONS = ("GK", "DF", "MF", "FW")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    def path(self, key: str) -> Path:
        return self.root / self.raw["data"][key]

    @property
    def raw_dir(self) -> Path:
        return self.path("raw_dir")

    @property
    def interim_dir(self) -> Path:
        return self.path("interim_dir")

    @property
    def processed_dir(self) -> Path:
        return self.path("processed_dir")

    @property
    def artifact_dir(self) -> Path:
        return self.path("artifact_dir")

    @property
    def competition_ids(self) -> list[str]:
        return list(self.raw["competition"]["ids"] or [])

    @property
    def study(self) -> dict[str, Any]:
        return self.raw["study"]

    @property
    def split(self) -> dict[str, Any]:
        return self.raw["split"]

    @property
    def cv(self) -> dict[str, Any]:
        return self.raw["cv"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw["runtime"]

    def date(self, section: str, key: str) -> dt.date | None:
        value = self.raw[section].get(key)
        return _as_date(value, f"{section}.{key}") if value is not None else None

    @property
    def test_start(self) -> dt.date:
        return self.date("split", "test_start")

    @property
    def transfer_start(self) -> dt.date:
        return self.date("study", "transfer_start")

    @property
    def transfer_end_exclusive(self) -> dt.date:
        return self.date("study", "transfer_end_exclusive")

    @property
    def min_minutes(self) -> int:
        return int(self.study["min_minutes"])

    def hash(self) -> str:
        return config_hash(self.raw)

    def require_pinned(self) -> None:
        """Q1, Q2 and T must be resolved before training or evaluation."""
        missing = []
        if not self.competition_ids:
            missing.append("competition.ids (Q1)")
        if (self.raw.get("fees") or {}).get("encoding") != "positive_is_paid_permanent":
            missing.append("fees.encoding (Q2)")
        for key in ("source_seasons", "season_ids", "transfer_start", "transfer_end_exclusive"):
            if not self.study.get(key):
                missing.append(f"study.{key}")
        if not self.split.get("test_start"):
            missing.append("split.test_start (Q3)")
        if missing:
            raise ConfigError("unresolved config values: " + ", ".join(missing))


def _as_date(value: Any, name: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(f"{name} is not an ISO date: {value!r}") from exc


def config_hash(raw: dict[str, Any]) -> str:
    canonical = json.dumps(raw, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    for section in ("data", "competition", "fees", "study", "split", "cv", "runtime"):
        if section not in raw:
            raise ConfigError(f"config missing section: {section}")
    cfg = Config(raw=raw, root=p.resolve().parent)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.min_minutes < 90:
        raise ConfigError("study.min_minutes must be >= 90")
    if cfg.study.get("include_cards") not in (True, False):
        raise ConfigError("study.include_cards must be true or false")
    if cfg.study.get("position_policy") != "historical_then_proxy":
        raise ConfigError("study.position_policy must be historical_then_proxy")
    n_folds = cfg.cv.get("n_folds")
    if not isinstance(n_folds, int) or n_folds < 1:
        raise ConfigError("cv.n_folds must be a positive integer")
    if not cfg.raw.get("fixture") and not 3 <= n_folds <= 5:
        raise ConfigError("cv.n_folds must be 3-5 outside fixture configs")
    for section, key in (
        ("study", "transfer_start"),
        ("study", "transfer_end_exclusive"),
        ("study", "prediction_as_of"),
        ("split", "test_start"),
    ):
        cfg.date(section, key)
    start, end, t = cfg.transfer_start, cfg.transfer_end_exclusive, cfg.test_start
    if start and end and t and not start < t < end:
        raise ConfigError("need transfer_start < split.test_start < transfer_end_exclusive")
