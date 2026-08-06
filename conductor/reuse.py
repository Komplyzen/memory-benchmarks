"""Wiring A: make run.py reuse a materialized store instead of ingesting.

run.py's ingest_question() checks an IngestionCheckpoint per question; if a
`_ingestion_{qid}.json` marks it complete (chunk_size matches) it SKIPS ingestion
and adopts the checkpoint's user_id for search. So to reuse a store we simply
pre-write those checkpoints pointing at the store's user_ids. run.py is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import db
from .paths import repo_root

# Must match benchmarks/longmemeval/run.py CHUNK_SIZE (pairs per ingest chunk).
_RUN_CHUNK_SIZE = 2


def load_store(store_artifact_id: str) -> dict[str, Any]:
    row = db.get_artifact(store_artifact_id)
    if not row:
        raise ValueError(f"no such artifact {store_artifact_id!r}")
    if row["layer"] != "memory_store":
        raise ValueError(f"artifact {store_artifact_id} is layer {row['layer']}, need memory_store")
    manifest = json.loads(row["origin_json"])
    if manifest.get("benchmark") == "browsecomp_plus":
        return {"manifest": manifest, "path": row["path"]}
    user_ids = json.loads((Path(row["path"]) / "user_ids.json").read_text())
    return {"manifest": manifest, "user_ids": user_ids["user_ids"],
            "per_question": user_ids.get("per_question", {})}


def checkpoint_dir(project_name: str) -> Path:
    # Mirrors run.py: output_dir = results/longmemeval/predicted_{project_name}
    return repo_root() / "results" / "longmemeval" / f"predicted_{project_name}"


def write_checkpoints(store_artifact_id: str, project_name: str) -> dict[str, Any]:
    """Pre-populate ingestion checkpoints so run.py skips ingestion for every
    question in the store. Returns store metadata for the origin story."""
    store = load_store(store_artifact_id)
    cp_dir = checkpoint_dir(project_name)
    cp_dir.mkdir(parents=True, exist_ok=True)
    per_q = store["per_question"]
    for qid, user_id in store["user_ids"].items():
        pairs = int(per_q.get(qid, {}).get("events", 0))
        (cp_dir / f"_ingestion_{qid}.json").write_text(json.dumps({
            "chunk_size": _RUN_CHUNK_SIZE,
            "user_id": user_id,
            "total_pairs_processed": pairs,
            "source": "conductor-reuse",
            "store_artifact": store_artifact_id,
        }, indent=2))
    m = store["manifest"]
    return {
        "store_artifact": store_artifact_id,
        "store_run_id": m.get("store_run_id"),
        "host": m.get("host"),
        "embedder": m.get("embedder"),
        "questions": len(store["user_ids"]),
        "dataset_path": (m.get("dataset") or {}).get("path"),
        "checkpoint_dir": str(cp_dir),
    }
