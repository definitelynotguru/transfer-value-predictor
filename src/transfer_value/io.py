"""Atomic artifact writes, run identity, and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from transfer_value import __version__


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    _atomic(path, lambda p: df.to_parquet(p, index=False))


def write_json(obj: Any, path: Path) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n"
    _atomic(path, lambda p: p.write_text(text))


def write_text(text: str, path: Path) -> None:
    _atomic(path, lambda p: p.write_text(text))


def write_joblib(obj: Any, path: Path) -> None:
    _atomic(path, lambda p: joblib.dump(obj, p))


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    return json.loads(path.read_text())


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    return pd.read_parquet(path)


def _json_default(o: Any) -> Any:
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash independent of parquet encoding details."""
    hashed = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def code_revision(root: Path) -> dict[str, Any]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": rev, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def environment_info(root: Path) -> dict[str, Any]:
    pkgs = ["pandas", "numpy", "scikit-learn", "matplotlib", "joblib", "pyarrow", "typer"]
    lock = root / "uv.lock"
    return {
        "python": platform.python_version(),
        "package_version": __version__,
        "dependencies": {p: metadata.version(p) for p in pkgs},
        "lockfile_sha256": sha256_file(lock) if lock.exists() else None,
    }


def run_id(source_hashes: dict[str, str], config_hash: str, lock_hash: str | None) -> str:
    """Deterministic: identical inputs give the same run ID."""
    payload = json.dumps(
        {"source": source_hashes, "config": config_hash, "lock": lock_hash, "v": __version__},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
