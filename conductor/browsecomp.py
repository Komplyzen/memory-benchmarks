from __future__ import annotations

import datetime as dt
import json
import secrets
from pathlib import Path

from . import db
from .paths import artifact_dir


def adopt_store(
    *,
    manifest_path: str,
    host: str,
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
    gold_qrels_path: str,
    artifact_id: str | None = None,
    note: str | None = None,
) -> dict:
    source = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(source.read_text())
    if manifest.get("benchmark") != "browsecomp_plus" or manifest.get("status") != "READY":
        raise ValueError("BrowseComp store manifest must be READY")
    aid = artifact_id or f"bcp-store-{secrets.token_hex(4)}"
    output_dir = artifact_dir("memory_store", aid)
    output_dir.mkdir(parents=True, exist_ok=True)
    adopted = {
        **manifest,
        "artifact_id": aid,
        "layer": "memory_store",
        "adopted": True,
        "adopted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host.rstrip("/"),
        "corpus": {**manifest["corpus"], "path": str(Path(corpus_path).expanduser().resolve())},
        "benchmark_files": {
            "queries": str(Path(queries_path).expanduser().resolve()),
            "qrels": str(Path(qrels_path).expanduser().resolve()),
            "gold_qrels": str(Path(gold_qrels_path).expanduser().resolve()),
        },
        "source_manifest": str(source),
        "note": note,
    }
    (output_dir / "manifest.json").write_text(json.dumps(adopted, indent=2) + "\n", encoding="utf-8")
    db.insert_artifact(
        artifact_id=aid,
        benchmark="browsecomp_plus",
        layer="memory_store",
        dataset_sha=adopted["corpus"].get("sha256"),
        path=str(output_dir),
        origin=adopted,
    )
    return adopted
