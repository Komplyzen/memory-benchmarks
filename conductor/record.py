"""Origin stories: the recorded facts of how the world was set up for a run.

The reuse decision (can a later run stand on this run's memories?) is made by an
agent reading these. So capture everything cheap and mechanical automatically --
never rely on the caller to remember -- and let the caller add freeform context on
top. The machine never interprets this; it just never loses it.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .paths import cache_dir, repo_root

_SECRET_HINT = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD")


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _git_state(cwd: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    head = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _run(["status", "--porcelain"])
    return {
        "commit": head,
        "branch": branch,
        "dirty": bool(porcelain) if porcelain is not None else None,
        "changed_files": porcelain.splitlines() if porcelain else [],
    }


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in env.items():
        if any(h in k.upper() for h in _SECRET_HINT):
            out[k] = (v[:6] + "…") if v else ""  # keep a prefix for identification
        else:
            out[k] = v
    return out


def _dataset_fingerprint(path: Optional[str]) -> Optional[dict[str, Any]]:
    """Size+mtime always; sha256 memoized (hashing a 277MB haystack once, not per run)."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return {"path": str(path), "exists": False}
    stat = p.stat()
    size, mtime = stat.st_size, int(stat.st_mtime)

    cache_file = cache_dir() / "dataset_hashes.json"
    cache: dict[str, Any] = {}
    if cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            cache = {}
    key = f"{p.resolve()}::{size}::{mtime}"
    sha = cache.get(key)
    if sha is None:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        cache[key] = sha
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
    return {"path": str(p.resolve()), "size": size, "mtime": mtime, "sha256": sha}


def _target(config: dict[str, Any], env_overrides: dict[str, str]) -> Optional[str]:
    """Where the memory platform under test lives (for the origin story + a per-target
    execution profile later). Precedence mirrors run.py: explicit host wins, else env."""
    host = config.get("mem0_host") or env_overrides.get("MEM0_HOST") or os.environ.get("MEM0_HOST")
    backend = config.get("backend")
    if host:
        return f"{host} ({backend})" if backend else host
    return backend


def capture_origin(
    *,
    benchmark: str,
    argv: list[str],
    config: dict[str, Any],
    env_overrides: dict[str, str],
    note: Optional[str] = None,
    describe: Optional[dict[str, str]] = None,
    dataset_path: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "captured_at_utc": _utc_now_iso(),
        "command": argv,
        "config": config,
        "target": _target(config, env_overrides),
        "git": {"benchmark_repo": _git_state(repo_root())},
        "env_overrides": _redact_env(env_overrides),
        "dataset": _dataset_fingerprint(dataset_path),
        "python": sys.version.split()[0],
        "note": note,
        "describe": describe or {},
    }
