"""The artifact shelf.

An artifact is a reusable rung of the eval ladder. v1 stores one layer:
`extracted_memories` -- the expensive LLM-extracted memory text, per benchmark
question, that a later run can re-ingest (via infer=false) instead of paying for
extraction again.

`adopt_longmemeval_dump()` imports an existing platform extraction run (raw
per-turn worker outputs) onto the shelf. It's the only adopter today; the storage
format below (manifest.json + memories.jsonl) is what every future producer/reader
targets.
"""

from __future__ import annotations

import datetime as _dt
import json
import secrets
from pathlib import Path
from typing import Any, Callable, Optional

from . import db
from .paths import artifact_dir
from .record import _dataset_fingerprint

LAYER = "extracted_memories"


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# --- question_id recovery ---
#
# run.py builds user_id = f"longmemeval_{question_id}_{run_id}". question_ids may
# themselves contain "_" (e.g. "0862e8bf_abs"), so we resolve against the dataset's
# real question_id set and match the LONGEST candidate, then treat the remainder as
# the (constant) run_id.


def _load_question_ids(dataset_path: str) -> set[str]:
    data = json.loads(Path(dataset_path).read_text())
    return {q["question_id"] for q in data}


def _split_user_id(user_id: str, qids: set[str]) -> Optional[tuple[str, str]]:
    if not user_id.startswith("longmemeval_"):
        return None
    rest = user_id[len("longmemeval_") :]
    best: Optional[str] = None
    for qid in qids:
        if rest.startswith(qid + "_") and (best is None or len(qid) > len(best)):
            best = qid
    if best is None:
        return None
    return best, rest[len(best) + 1 :]  # (question_id, run_id)


# --- adopt a platform extraction dump ---


def _iter_cycles(source: Path) -> list[Path]:
    cycles = sorted(source.glob("cycle_*"))
    return [c for c in cycles if (c / "extraction_calls").is_dir()]


def adopt_longmemeval_dump(
    *,
    source_dir: str,
    dataset_path: str,
    artifact_id: Optional[str] = None,
    note: Optional[str] = None,
    max_questions: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Convert a per-turn extraction dump into a shelf artifact.

    Joins extraction_calls (memory text) to queue_payloads (user_id + timestamp) by
    event_id, groups by canonical question_id, and writes manifest.json +
    memories.jsonl. Returns the manifest dict.
    """
    say = progress or (lambda _m: None)
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dir not found: {source}")
    qids = _load_question_ids(dataset_path)
    cycles = _iter_cycles(source)
    if not cycles:
        raise FileNotFoundError(f"no cycle_*/extraction_calls under {source}")

    # question_id -> list of EVENTS (each = one original add-event with its source
    # messages + recorded extraction). This is the faithful replay feed.
    per_q: dict[str, list[dict[str, Any]]] = {}
    seen_questions: set[str] = set()
    seen_events: set[str] = set()  # dedup a turn even if it recurs across cycles/reaps
    extraction_model: Optional[str] = None
    stats = {"extraction_files": 0, "payload_missing": 0, "parse_failed": 0,
             "empty_turns": 0, "memories": 0, "events": 0,
             "unmapped_user_ids": 0, "duplicate_events": 0}
    run_ids: set[str] = set()

    for cyc in cycles:
        ec_dir = cyc / "extraction_calls"
        qp_dir = cyc / "queue_payloads"
        say(f"scanning {cyc.name} ...")
        for ec_path in ec_dir.glob("*.json"):
            event_id = ec_path.name.split(".", 1)[0]
            stats["extraction_files"] += 1
            if event_id in seen_events:
                stats["duplicate_events"] += 1
                continue
            seen_events.add(event_id)
            qp_path = qp_dir / f"{event_id}.json"
            if not qp_path.is_file():
                stats["payload_missing"] += 1
                continue
            try:
                payload = json.loads(qp_path.read_text())
                user_id = payload["filters"]["user_id_str"]
                timestamp = payload.get("timestamp")
                messages = payload.get("messages")
                order_key = payload.get("sqs_sent_at_ns") or timestamp or 0
                rec = json.loads(ec_path.read_text())
                mems = json.loads(rec["response"]["content"])["memory"]
                if extraction_model is None:
                    extraction_model = rec.get("model")
            except Exception:
                stats["parse_failed"] += 1
                continue

            split = _split_user_id(user_id, qids)
            if split is None:
                stats["unmapped_user_ids"] += 1
                continue
            qid, run_id = split

            if qid not in seen_questions:
                # Cap distinct questions for smoke adopts.
                if max_questions is not None and len(seen_questions) >= max_questions:
                    continue
                seen_questions.add(qid)
            run_ids.add(run_id)

            if not mems:
                stats["empty_turns"] += 1
            per_q.setdefault(qid, []).append({
                "event_id": event_id,
                "timestamp": timestamp,
                "order_key": order_key,
                "messages": messages,
                "memories": mems,
            })
            stats["events"] += 1
            stats["memories"] += len(mems)

    if not per_q:
        raise RuntimeError("no memories mapped -- check dataset_path matches the dump")

    # Original ingest order per question (send time; event id as stable tiebreak).
    for qid, evts in per_q.items():
        evts.sort(key=lambda e: (e["order_key"], e["event_id"]))

    aid = artifact_id or f"emx-{secrets.token_hex(4)}"
    out_dir = artifact_dir(LAYER, aid)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id_suffix = next(iter(run_ids))

    # events.jsonl = the replay feed: one line per add-event, grouped by question,
    # in original order. materialize streams this into /replay.
    say(f"writing {stats['events']} events / {stats['memories']} memories -> {out_dir}")
    with (out_dir / "events.jsonl").open("w") as ef:
        for qid in sorted(per_q):
            for e in per_q[qid]:
                ef.write(json.dumps({
                    "question_id": qid,
                    "event_id": e["event_id"],
                    "timestamp": e["timestamp"],
                    "messages": e["messages"],
                    "memories": e["memories"],
                }) + "\n")

    # memories.jsonl = flattened per-question view (inspection + counts).
    with (out_dir / "memories.jsonl").open("w") as mf:
        for qid in sorted(per_q):
            flat = [
                {"text": m.get("text"), "timestamp": e["timestamp"],
                 "attributed_to": m.get("attributed_to"), "source_event_id": e["event_id"]}
                for e in per_q[qid] for m in e["memories"]
            ]
            mf.write(json.dumps({
                "question_id": qid,
                "original_user_id": f"longmemeval_{qid}_{run_id_suffix}",
                "memory_count": len(flat),
                "memories": flat,
            }) + "\n")

    dataset_fp = _dataset_fingerprint(dataset_path) or {}
    source_run_json = _read_run_json(source)
    manifest = {
        "artifact_id": aid,
        "benchmark": "longmemeval",
        "layer": LAYER,
        "schema": 2,
        "replay_feed": "events.jsonl",
        "created_at_utc": _utc_now_iso(),
        "adopted": True,
        "source_dir": str(source),
        "dataset": dataset_fp,
        "question_count": len(per_q),
        "event_count": stats["events"],
        "memory_count": stats["memories"],
        "extraction_model": extraction_model,
        "run_ids": sorted(run_ids),
        "stats": stats,
        "caveats": [
            "This artifact holds RAW per-turn extraction output (pre-persistence). "
            "`conductor materialize` replays events.jsonl through the real infer=True "
            "pipeline, so the materialized store gets faithful hash-dedup, BM25 fields, "
            "and entity linking -- the store, not this raw artifact, is the reusable base.",
            "Extractions came from the platform; a store materialized on the OSS backend "
            "is faithful to OSS persistence semantics given platform-quality extractions, "
            "not a byte-replica of the original platform store.",
            "Any linked_memory_ids from the source run point at defunct platform UUIDs "
            "and are intentionally dropped.",
        ],
        "note": note,
        "source_run_json": source_run_json,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    db.insert_artifact(
        artifact_id=aid,
        benchmark="longmemeval",
        layer=LAYER,
        dataset_sha=dataset_fp.get("sha256"),
        path=str(out_dir),
        origin=manifest,
    )
    return manifest


def _read_run_json(source: Path) -> Optional[dict[str, Any]]:
    for candidate in ("manifests/run.json", "run.json"):
        p = source / candidate
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def load_manifest(artifact_id: str) -> Optional[dict[str, Any]]:
    row = db.get_artifact(artifact_id)
    if not row:
        return None
    mp = Path(row["path"]) / "manifest.json"
    return json.loads(mp.read_text()) if mp.is_file() else None
