"""Load verified raw files, normalize, and write typed interim Parquet plus a manifest."""

from __future__ import annotations

import shutil
from pathlib import Path

from transfer_value.config import RAW_FILES, Config
from transfer_value.dataset import canonical_tables
from transfer_value.io import (
    environment_info,
    frame_fingerprint,
    write_json,
    write_parquet,
)
from transfer_value.source import read_source_manifest, sha256_file, verify_raw_files


def import_manual_snapshot(cfg: Config, input_dir: Path) -> None:
    """Copy a manually downloaded snapshot into raw_dir after checking pinned hashes."""
    pinned = cfg.raw["source"].get("sha256") or {}
    for name in RAW_FILES.values():
        src = input_dir / name
        if not src.exists():
            raise FileNotFoundError(f"manual snapshot missing {src}")
        digest = sha256_file(src)
        if pinned.get(name) and pinned[name] != digest:
            raise ValueError(f"{name}: sha256 {digest} != pinned {pinned[name]}")
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES.values():
        dest = cfg.raw_dir / name
        if dest.exists() and sha256_file(dest) != sha256_file(input_dir / name):
            raise ValueError(f"{dest} exists with different bytes; refusing to replace")
        shutil.copy2(input_dir / name, dest)
    manifest = input_dir / "SOURCE.md"
    if manifest.exists() and not (cfg.raw_dir / "SOURCE.md").exists():
        shutil.copy2(manifest, cfg.raw_dir / "SOURCE.md")


def verified_source_hashes(cfg: Config) -> dict[str, str]:
    """Hashes of required raw files, checked against SOURCE.md and config pins."""
    hashes = verify_raw_files(cfg.raw_dir, list(RAW_FILES.values()))
    pinned = cfg.raw["source"].get("sha256") or {}
    for name, digest in hashes.items():
        if pinned.get(name) and pinned[name] != digest:
            raise ValueError(f"{name}: sha256 {digest} != config pin {pinned[name]}")
    return hashes


def run_ingest(cfg: Config, input_dir: Path | None = None) -> dict:
    if input_dir is not None:
        import_manual_snapshot(cfg, input_dir)
    hashes = verified_source_hashes(cfg)
    tables, stats = canonical_tables(cfg, cfg.raw_dir)
    fingerprints = {}
    for name, df in tables.items():
        write_parquet(df, cfg.interim_dir / f"{name}.parquet")
        fingerprints[name] = frame_fingerprint(df)
    entries = {e["file"]: e for e in read_source_manifest(cfg.raw_dir / "SOURCE.md")}
    manifest = {
        "stage": "ingest",
        "source_hashes": hashes,
        "source_entries": {k: entries[k] for k in hashes},
        "config_hash": cfg.hash(),
        "competition_ids": cfg.competition_ids,
        "row_counts": {k: len(v) for k, v in tables.items()},
        "table_fingerprints": fingerprints,
        "cleaning_stats": stats,
        "environment": environment_info(cfg.root),
    }
    write_json(manifest, cfg.interim_dir / "manifest.json")
    return manifest
