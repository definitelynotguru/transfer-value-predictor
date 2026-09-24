"""Download the pinned Transfermarkt community snapshot and record per-file SHA-256.

This is the only network step in the project. Each file is streamed to a temporary path,
hashed, checked against any hash pinned in config.yaml, and only then moved into place.
An existing file is never silently replaced: a mismatch fails unless --refresh is given.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transfer_value.source import read_source_manifest, write_source_manifest  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest_dir: Path) -> tuple[Path, dict]:
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=".download-")
    tmp = Path(tmp_name)
    # The public bucket rejects urllib's default User-Agent with 403.
    req = urllib.request.Request(url, headers={"User-Agent": "transfer-value-fetch/0.1"})
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=120) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            shutil.copyfileobj(resp, out, length=1 << 20)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp, headers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Allow replacing files whose bytes differ from the pinned/local hash.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    src = cfg["source"]
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    pinned: dict = src.get("sha256") or {}

    entries = []
    for name in src["files"]:
        url = f"{src['base_url'].rstrip('/')}/{name}"
        dest = raw_dir / name
        print(f"fetch {url}")
        tmp, headers = download(url, raw_dir)
        digest = sha256_file(tmp)
        expected = pinned.get(name)
        if expected and expected != digest and not args.refresh:
            tmp.unlink()
            print(f"  FAIL sha256 mismatch for {name}: pinned {expected}, got {digest}")
            return 1
        if dest.exists() and sha256_file(dest) != digest and not args.refresh:
            tmp.unlink()
            print(f"  FAIL {dest} exists with different bytes; rerun with --refresh")
            return 1
        os.replace(tmp, dest)
        entries.append(
            {
                "file": name,
                "url": url,
                "bytes": dest.stat().st_size,
                "sha256": digest,
                "acquired_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "upstream_last_modified": headers.get("last-modified"),
                "upstream_etag": headers.get("etag", "").strip('"') or None,
            }
        )
        print(f"  ok {dest.stat().st_size:,} bytes sha256={digest}")

    # Merge so a second config (e.g. followup.yaml) adds its files without dropping others.
    manifest = raw_dir / "SOURCE.md"
    fetched = {e["file"] for e in entries}
    if manifest.exists():
        entries += [e for e in read_source_manifest(manifest) if e["file"] not in fetched]
    write_source_manifest(manifest, entries, src)
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
