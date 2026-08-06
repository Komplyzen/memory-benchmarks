"""Instruments for auditing a materialized store. EVIDENCE, NOT VERDICT.

These checks catch known, recurring mis-wirings (timestamps not passed =>
memories dated at ingest time; enrichment skipped => empty entity store;
partial stores). A red flag is strong evidence something is wrong. All-clear
means only "these specific instruments saw nothing" -- the failure space is
open and novel mistakes will not be on this list. The operator agent's
open-ended review (experiment intent + expected platform behavior) is the
load-bearing judgment; this module just feeds it.
"""

from __future__ import annotations

import datetime as _dt
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import db

# Memories dated within this many days of "now" are treated as ingest-time
# dates, i.e. evidence that dataset timestamps were NOT honored.
_RECENT_DAYS = 14


def _get(url: str, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url: str, body: dict[str, Any], timeout: float = 60.0) -> Any:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def verify_store(
    store_artifact_id: str,
    *,
    sample: int = 8,
    qdrant: str = "http://localhost:6333",
    seed: int = 7,
) -> dict[str, Any]:
    """Returns {passed: bool, checks: [{name, ok, detail}]}; raises on missing store."""
    row = db.get_artifact(store_artifact_id)
    if not row:
        raise ValueError(f"no such artifact {store_artifact_id!r}")
    if row["layer"] != "memory_store":
        raise ValueError(f"{store_artifact_id} is layer {row['layer']}, need memory_store")
    manifest = json.loads(row["origin_json"])
    host = str(manifest.get("host", "")).rstrip("/")
    user_map = json.loads((Path(row["path"]) / "user_ids.json").read_text())
    user_ids: dict[str, str] = user_map["user_ids"]
    per_q: dict[str, Any] = user_map.get("per_question", {})

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # 1. Store reachable
    try:
        health = _get(f"{host}/health", timeout=10)
        check("platform reachable", True, f"{host} ({health.get('embedder')})")
    except Exception as e:
        check("platform reachable", False, f"{host}: {str(e)[:120]}")
        return {"passed": False, "checks": checks}

    # 2. Sampled questions: memories exist, counts match, timestamps honored
    rng = random.Random(seed)
    qids = rng.sample(sorted(user_ids), min(sample, len(user_ids)))
    now = _dt.datetime.now(_dt.timezone.utc)
    missing, count_mismatch, recent_dated, sampled_memories = [], [], [], 0
    for qid in qids:
        uid = user_ids[qid]
        listed = _get(f"{host}/memories?user_id={urllib.parse.quote(uid)}")
        mems = listed.get("results", listed if isinstance(listed, list) else [])
        sampled_memories += len(mems)
        if not mems:
            missing.append(qid)
            continue
        expected = (per_q.get(qid) or {}).get("stored_memories")
        if expected is not None and len(mems) != expected:
            count_mismatch.append(f"{qid}: {len(mems)} vs manifest {expected}")
        # Timestamp fidelity: dataset timelines are historical; a store whose
        # memories are all dated ~today was ingested WITHOUT dataset timestamps.
        dates = [m.get("created_at") or m.get("metadata", {}).get("created_at") for m in mems]
        dates = [d for d in dates if d]
        if dates:
            recent = 0
            for d in dates:
                try:
                    dt = _dt.datetime.fromisoformat(str(d).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_dt.timezone.utc)
                    if (now - dt).days <= _RECENT_DAYS:
                        recent += 1
                except Exception:
                    pass
            if recent > len(dates) * 0.5:
                recent_dated.append(qid)

    check("memories present for sampled questions",
          not missing, f"sampled {len(qids)} questions, {sampled_memories} memories"
          + (f"; MISSING: {missing}" if missing else ""))
    check("per-question counts match manifest",
          not count_mismatch, "; ".join(count_mismatch) if count_mismatch else "all sampled match")
    check("dataset timestamps honored (not ingest-dated)",
          not recent_dated,
          f"questions dated ~today: {recent_dated}" if recent_dated
          else "sampled memories carry historical dataset dates")

    # 3. Entity store populated (the silently-amputated-signal failure mode)
    try:
        cols = _get(f"{qdrant.rstrip('/')}/collections")["result"]["collections"]
        names = [c["name"] for c in cols]
        ent = [n for n in names if n.endswith("_entities")]
        ent_counts = {}
        for n in ent:
            ent_counts[n] = _get(f"{qdrant.rstrip('/')}/collections/{n}")["result"].get(
                "points_count", 0)
        ok = any(v > 0 for v in ent_counts.values())
        check("entity store populated", ok,
              ", ".join(f"{k}={v}" for k, v in ent_counts.items()) or "no *_entities collection")
    except Exception as e:
        check("entity store populated", False, f"qdrant check failed: {str(e)[:120]}")

    # 4. Search sanity: one real query returns hits from the store
    try:
        uid = user_ids[qids[0]]
        s = _post(f"{host}/search", {"query": "user preferences", "user_id": uid, "limit": 3})
        hits = s.get("results", s if isinstance(s, list) else [])
        check("search returns hits", len(hits) > 0, f"{len(hits)} hits for sampled user")
    except Exception as e:
        check("search returns hits", False, str(e)[:120])

    return {"passed": all(c["ok"] for c in checks), "checks": checks}
