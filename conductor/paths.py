"""Filesystem layout for conductor.

Two roots, deliberately separable (contract #1 -- one data root, so migrating to
a VM is `rsync the root; point config at it`):

- DB path: defaults to `<repo>/evals.db`, the exact file the Next.js web UI reads
  (`src/lib/db.ts`). Pinned to the repo by default so CLI and UI share ONE ledger.
  Override with CONDUCTOR_DB only if you know what you're doing.
- data root: everything else conductor writes (per-run dirs, logs, origin stories,
  and later the artifact shelf + exports). Defaults to the repo root; override with
  CONDUCTOR_DATA_ROOT to relocate all conductor state (e.g. a big data disk).
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """The memory-benchmarks checkout root (parent of the conductor/ package)."""
    return Path(__file__).resolve().parent.parent


def db_path() -> Path:
    env = os.environ.get("CONDUCTOR_DB")
    if env:
        return Path(env).expanduser().resolve()
    # Match the web UI (path.join(process.cwd(), "evals.db")) so we share a ledger.
    return repo_root() / "evals.db"


def data_root() -> Path:
    env = os.environ.get("CONDUCTOR_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root()


def runs_dir() -> Path:
    """Per-run working dirs: launch spec, origin story, log."""
    return data_root() / "conductor_state" / "runs"


def cache_dir() -> Path:
    """Small memoized computations (e.g. dataset hashes)."""
    return data_root() / "conductor_state" / "cache"


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def shelf_dir() -> Path:
    """The artifact shelf root (per-layer subdirs beneath)."""
    return data_root() / "conductor_state" / "shelf"


def artifact_dir(layer: str, artifact_id: str) -> Path:
    return shelf_dir() / layer / artifact_id
