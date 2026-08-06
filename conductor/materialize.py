"""Materialize a shelf artifact into a live, faithful memory store.

Replays an `extracted_memories` artifact's events.jsonl through the server's
/replay endpoint (the real infer=True pipeline with recorded extractions), which
populates the target platform with dedup + BM25 + entity linking -- a store
identical to one a normal ingestion run would build. The populated store is
recorded as a new `memory_store` shelf artifact (its own reusable rung).

Events within a question replay in original order (dedup depends on order);
questions run in parallel.
"""

from __future__ import annotations

import datetime as _dt
import json
import secrets
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from . import db, preflight
from .paths import artifact_dir

STORE_LAYER = "memory_store"


class MaterializeAborted(RuntimeError):
    """Raised when materialize halts mid-run (e.g. embedder creds expired).
    Progress is checkpointed; re-running with the same --store-id resumes."""


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _post(host: str, ep: str, body: dict[str, Any], timeout: float = 120.0) -> Any:
    req = urllib.request.Request(
        f"{host}{ep}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _health(host: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def materialize(
    *,
    artifact_id: str,
    host: str,
    store_id: Optional[str] = None,
    max_questions: Optional[int] = None,
    question_ids: Optional[list[str]] = None,
    max_workers: int = 8,
    note: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    host = host.rstrip("/")

    src = db.get_artifact(artifact_id)
    if not src:
        raise ValueError(f"no such artifact {artifact_id!r}")
    if src["layer"] != "extracted_memories":
        raise ValueError(f"artifact {artifact_id} is layer {src['layer']}, need extracted_memories")
    src_manifest = json.loads(src["origin_json"])
    events_path = Path(src["path"]) / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"artifact has no events.jsonl (re-adopt with schema 2): {events_path}")

    # Fail loud if the target embedder/platform is unusable -- never proceed blind.
    preflight.require_ok({"backend": "oss", "mem0_host": host}, {})

    # Group events by question, preserving file order (already ordered per question).
    say("loading replay feed ...")
    by_q: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    with events_path.open() as f:
        for line in f:
            e = json.loads(line)
            by_q.setdefault(e["question_id"], []).append(e)
    if question_ids is not None:
        want = set(question_ids)
        by_q = OrderedDict((q, v) for q, v in by_q.items() if q in want)
        missing = want - set(by_q)
        if missing:
            raise ValueError(f"{len(missing)} requested question_ids not in artifact: {sorted(missing)[:5]}")
    if max_questions is not None:
        by_q = OrderedDict(list(by_q.items())[:max_questions])

    store_run_id = store_id or f"ms{secrets.token_hex(3)}"
    aid = store_run_id
    out_dir = artifact_dir(STORE_LAYER, aid)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    # Resume: skip questions already completed in a prior run of this store.
    results: list[dict[str, Any]] = []
    completed: set[str] = set()
    if progress_path.is_file():
        for line in progress_path.read_text().splitlines():
            try:
                r = json.loads(line)
                completed.add(r["question_id"])
                results.append(r)
            except Exception:
                pass
    todo = OrderedDict((q, v) for q, v in by_q.items() if q not in completed)
    if completed:
        say(f"resuming: {len(completed)} already done, {len(todo)} to go")
    say(f"materializing {len(todo)}/{len(by_q)} questions -> store_run_id={store_run_id} @ {host}")

    abort = threading.Event()
    abort_reason = {"summary": "", "hint": ""}
    lock = threading.Lock()

    def do_question(qid: str, events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if abort.is_set():
            return None
        user_id = f"longmemeval_{qid}_{store_run_id}"
        try:
            res = _post(host, "/replay_bulk", {
                "user_id": user_id,
                "events": [{"messages": e["messages"], "memories": e["memories"],
                            "timestamp": e.get("timestamp")} for e in events],
            }, timeout=1800)
        except urllib.error.HTTPError as ex:
            body = ex.read().decode(errors="replace")
            summary, hint = preflight._classify(ex.code, body)
            # Embedder/credential failure => halt the whole run loudly (don't burn
            # through the rest marking them all failed). User refreshes + resumes.
            if any(k in summary for k in ("credential", "endpoint", "access")):
                abort_reason.update(summary=summary, hint=hint)
                abort.set()
            raise RuntimeError(f"{qid}: {summary}") from None
        rec = {"question_id": qid, "user_id": user_id,
               "events": len(events), "stored_memories": res.get("stored", 0)}
        with lock:  # append to the resume checkpoint immediately
            with progress_path.open("a") as pf:
                pf.write(json.dumps(rec) + "\n")
        return rec

    errors: list[dict[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(do_question, qid, evts): qid for qid, evts in todo.items()}
        for fut in as_completed(futs):
            qid = futs[fut]
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception as ex:
                errors.append({"question_id": qid, "error": str(ex)[:200]})
            done += 1
            if done % 25 == 0 or done == len(todo):
                say(f"  {done}/{len(todo)} this run ({len(results)}/{len(by_q)} total)")

    if abort.is_set():
        raise MaterializeAborted(
            f"MATERIALIZE HALTED @ {host}: {abort_reason['summary']}. "
            f"{abort_reason['hint']} Progress saved ({len(results)}/{len(by_q)} done); "
            f"refresh creds and re-run with --store-id {store_run_id} to resume."
        )
    if not results:
        raise RuntimeError(f"materialize produced no stores; errors: {errors[:3]}")

    total_stored = sum(r["stored_memories"] for r in results)
    total_events = sum(r["events"] for r in results)
    health = _health(host)
    user_map = {r["question_id"]: r["user_id"] for r in sorted(results, key=lambda x: x["question_id"])}
    (out_dir / "user_ids.json").write_text(json.dumps({
        "store_run_id": store_run_id,
        "user_ids": user_map,
        "per_question": {r["question_id"]: {"events": r["events"], "stored_memories": r["stored_memories"]}
                         for r in results},
    }, indent=2))

    manifest = {
        "artifact_id": aid,
        "benchmark": "longmemeval",
        "layer": STORE_LAYER,
        "schema": 1,
        "created_at_utc": _utc_now_iso(),
        "source_artifact": artifact_id,
        "store_run_id": store_run_id,
        "host": host,
        "embedder": health.get("embedder"),
        "llm": health.get("llm"),
        "question_count": len(results),
        "event_count": total_events,
        "stored_memory_count": total_stored,
        "raw_memory_count": src_manifest.get("memory_count"),
        # Only meaningful for a FULL materialize (subset raw counts aren't the artifact total).
        "dedup_dropped": (src_manifest.get("memory_count") or 0) - total_stored
        if (max_questions is None and question_ids is None) else None,
        "extraction_model": src_manifest.get("extraction_model"),
        "dataset": src_manifest.get("dataset"),
        "errors": errors,
        "note": note,
        "reuse_hint": "faithful store (real pipeline: dedup + BM25 + entity linking); "
                      "reuse via `conductor start longmemeval --reuse " + aid + "`.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    db.insert_artifact(
        artifact_id=aid,
        benchmark="longmemeval",
        layer=STORE_LAYER,
        dataset_sha=(src_manifest.get("dataset") or {}).get("sha256"),
        path=str(out_dir),
        origin=manifest,
    )
    return manifest
