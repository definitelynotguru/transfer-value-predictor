"""Source provenance: SOURCE.md manifest writer/reader and hash verification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_ENTRIES_START = "<!-- source-entries"
_ENTRIES_END = "-->"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_source_manifest(path: Path, entries: list[dict], source_cfg: dict) -> None:
    rows = "\n".join(
        f"| `{e['file']}` | {e['bytes']:,} | `{e['sha256']}` | {e['acquired_utc']} "
        f"| {e.get('upstream_last_modified') or 'n/a'} |"
        for e in sorted(entries, key=lambda e: e["file"])
    )
    text = f"""# Data source

Raw files are **not** committed. This manifest identifies the exact bytes used.

- Dataset: Transfermarkt community datasets by David Cariboo
  ([dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets),
  [Kaggle mirror](https://www.kaggle.com/datasets/davidcariboo/player-scores)).
- Underlying data: [Transfermarkt](https://www.transfermarkt.com/). All rights remain with
  Transfermarkt and the dataset maintainers.
- Base URL: `{source_cfg["base_url"]}`
- Snapshot note: {source_cfg.get("snapshot_note", "n/a")}

The upstream URLs are mutable. A matching SHA-256 proves you have the same bytes; it does not
guarantee those bytes remain downloadable. Keep a private copy if exact reproduction matters.

| File | Bytes | SHA-256 | Acquired (UTC) | Upstream Last-Modified |
|------|------:|---------|----------------|------------------------|
{rows}

{_ENTRIES_START}
{json.dumps(sorted(entries, key=lambda e: e["file"]), indent=2)}
{_ENTRIES_END}
"""
    path.write_text(text)


def read_source_manifest(path: Path) -> list[dict]:
    text = path.read_text()
    m = re.search(re.escape(_ENTRIES_START) + r"\n(.*?)\n" + re.escape(_ENTRIES_END), text, re.S)
    if not m:
        raise ValueError(f"{path} has no machine-readable source entries")
    return json.loads(m.group(1))


def verify_raw_files(raw_dir: Path, required: list[str]) -> dict[str, str]:
    """Return {file: sha256}; fail if a required file is missing or differs from SOURCE.md."""
    manifest_path = raw_dir / "SOURCE.md"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} missing; run scripts/fetch_data.py first")
    recorded = {e["file"]: e["sha256"] for e in read_source_manifest(manifest_path)}
    hashes = {}
    for name in required:
        p = raw_dir / name
        if not p.exists():
            raise FileNotFoundError(f"required raw file missing: {p}")
        if name not in recorded:
            raise ValueError(f"{name} not recorded in SOURCE.md")
        digest = sha256_file(p)
        if digest != recorded[name]:
            raise ValueError(f"{name} sha256 {digest} != SOURCE.md {recorded[name]}")
        hashes[name] = digest
    return hashes
