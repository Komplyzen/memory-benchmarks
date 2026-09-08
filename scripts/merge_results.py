#!/usr/bin/env python3
"""
Merge shard legs of one backend into a unified result file (spec-benchmark-ci CAP-6)
=====================================================================================

    python -m scripts.merge_results --benchmark locomo --backend kg-full \
        --expected-shards 0,1,2 [--allow-partial] --out FILE --run-id ID --run-attempt N DIR...

Each DIR is a leg's ``predicted_<project>/`` directory: the runner's per-question
JSON files plus the ``leg_meta.json`` sidecar bench.sh writes (see
benchmarks/common/bench_common.py for its shape).

Refuses (exit 2) when: a leg has zero evaluations; a leg's backend differs from
--backend; a shard is missing and --allow-partial is not set; legs differ in
answerer_model, judge_model, cutoffs, harness_sha, or kg_variant; a question_id
appears in two legs.

Writes the unified file in the runner's own shape ({metadata, metrics_by_cutoff,
evaluations}); metrics are recomputed with the benchmark's own function
(compute_locomo_metrics / compute_longmemeval_metrics). Metadata adds
github_run_id, run_attempt, legs, missing_shards, harness_sha (git rev-parse
HEAD of the fork), embedding_model, kg_* ingest counters summed across legs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any

from benchmarks.common.bench_common import (
    BACKEND_VARIANT,
    BenchError,
    answer_judge_calls,
    cutoffs_for_backend,
    harness_sha,
    labels_for,
    load_leg,
    normalize_backends,
    normalize_cutoffs,
)
from benchmarks.common.utils import save_result_json

SUMMABLE_KG_KEYS = (
    "kg_ingest_sessions", "kg_ingest_llm_calls", "kg_ingest_prompt_tokens", "kg_ingest_completion_tokens",
    "kg_ingest_cached_prompt_tokens", "kg_ingest_cache_write_tokens", "kg_ingest_tool_calls_total",
    "kg_ingest_max_tool_calls_hit", "kg_ingest_llm_failures", "kg_nodes_created",
    "none_questions", "none_questions_truncated", "none_sessions_dropped_total", "none_chars_dropped_total",
)
SUMMABLE_KG_DICTS = ("kg_ingest_tool_calls_by_name", "kg_ingest_tool_errors_by_name",
                     "kg_ingest_tool_calls_per_llm_call_histogram", "kg_guardrails")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=["locomo", "longmemeval"])
    p.add_argument("--backend", required=True)
    p.add_argument("--expected-shards", required=True, help="comma list of shard indices that should be present")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-attempt", type=int, required=True)
    p.add_argument("dirs", nargs="+")
    return p.parse_args()


def leg_summary(leg: dict[str, Any]) -> dict[str, Any]:
    meta = leg["meta"]
    kg_meta = meta.get("kg_meta") or {}
    return {
        "shard": meta.get("shard"),
        "dir": os.path.basename(leg["dir"].rstrip("/")),
        "evaluations": len(leg["evaluations"]),
        "wall_seconds": meta.get("wall_seconds"),
        "phase_failed": meta.get("phase_failed"),
        "kg_commit": meta.get("kg_commit") or kg_meta.get("kg_commit"),
        "kg_ingest_llm_calls": kg_meta.get("kg_ingest_llm_calls"),
        "answer_judge_calls": answer_judge_calls(leg["evaluations"]),
        "nodes_created": kg_meta.get("kg_nodes_created") or (kg_meta.get("kg_ingest_tool_calls_by_name") or {}).get("kg_node"),
        "embed_failures": meta.get("embed_failures"),
        "versions": meta.get("versions"),
    }


def merge_kg_meta(legs: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for leg in legs:
        km = leg["meta"].get("kg_meta") or {}
        for k, v in km.items():
            if k in SUMMABLE_KG_KEYS and isinstance(v, (int, float)):
                merged[k] = merged.get(k, 0) + v
            elif k in SUMMABLE_KG_DICTS and isinstance(v, dict):
                c = Counter(merged.get(k, {}))
                c.update({kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))})
                merged[k] = dict(c)
            elif k not in merged:
                merged[k] = v
    p, c = merged.get("kg_ingest_prompt_tokens"), merged.get("kg_ingest_cached_prompt_tokens")
    if p:
        merged["kg_ingest_cache_hit_ratio"] = round((c or 0) / p, 4)
    return merged


def main() -> int:
    args = parse_args()
    try:
        backend = normalize_backends(args.backend)[0]
        expected = sorted({int(x) for x in args.expected_shards.split(",") if x.strip()})
        legs = [load_leg(d) for d in args.dirs]

        for leg in legs:
            meta = leg["meta"]
            leg_backend = normalize_backends(str(meta.get("backend", "")))[0] if meta.get("backend") else None
            if leg_backend != backend:
                raise BenchError(f"{leg['dir']}: leg backend {meta.get('backend')!r} != --backend {backend!r}", 2)
            if not leg["evaluations"]:
                raise BenchError(f"{leg['dir']}: zero evaluations (phase_failed={meta.get('phase_failed')!r}); refusing to merge", 2)

        present = sorted({int(leg["meta"].get("shard")) for leg in legs if leg["meta"].get("shard") is not None})
        missing = [s for s in expected if s not in present]
        unexpected = [s for s in present if s not in expected]
        if unexpected:
            raise BenchError(f"legs for shards {unexpected} not in --expected-shards {expected}", 2)
        if missing and not args.allow_partial:
            raise BenchError(f"missing shards {missing} (present {present}); pass --allow-partial to merge anyway", 2)

        # Consistency across legs
        def key(leg: dict[str, Any]) -> tuple:
            m = leg["meta"]
            models = m.get("models") or {}
            cut = tuple(normalize_cutoffs(m.get("cutoffs") or [])) if m.get("cutoffs") else ()
            variant = (m.get("kg_meta") or {}).get("kg_variant") or BACKEND_VARIANT.get(backend)
            return (models.get("answerer_model"), models.get("judge_model"), cut, m.get("harness_sha"), variant)

        keys = {key(l) for l in legs}
        if len(keys) > 1:
            lines = "\n".join(f"  {os.path.basename(l['dir'])}: answerer={key(l)[0]} judge={key(l)[1]} cutoffs={list(key(l)[2])} harness={key(l)[3]} variant={key(l)[4]}" for l in legs)
            raise BenchError(f"legs differ in answerer_model/judge_model/cutoffs/harness_sha/kg_variant:\n{lines}", 2)
        answerer, judge, cutoffs_t, leg_harness, variant = next(iter(keys))
        cutoffs = list(cutoffs_t) if cutoffs_t else []
        if not cutoffs:
            raise BenchError("leg_meta.json has no cutoffs", 2)

        evaluations: list[dict[str, Any]] = []
        seen: dict[str, str] = {}
        for leg in sorted(legs, key=lambda l: l["meta"].get("shard", 0)):
            for e in leg["evaluations"]:
                qid = e["question_id"]
                if qid in seen:
                    raise BenchError(f"question_id {qid} appears in both {seen[qid]} and {leg['dir']}", 2)
                seen[qid] = leg["dir"]
                evaluations.append(e)

        eval_cutoffs = cutoffs_for_backend(backend, cutoffs)
        if args.benchmark == "locomo":
            from benchmarks.locomo.run import compute_locomo_metrics
            metrics = compute_locomo_metrics(evaluations, eval_cutoffs)
            group_key, groups = "categories", sorted({e.get("category") for e in evaluations if e.get("category") is not None})
        else:
            from benchmarks.longmemeval.run import compute_longmemeval_metrics
            metrics = compute_longmemeval_metrics(evaluations, eval_cutoffs)
            group_key, groups = "question_types", sorted({e.get("question_type", "unknown") for e in evaluations})

        first = legs[0]["meta"]
        models = first.get("models") or {}
        kg_meta = merge_kg_meta(legs)
        mem0 = first.get("mem0") or {}
        sha = harness_sha() or leg_harness

        metadata: dict[str, Any] = {
            "benchmark": args.benchmark,
            "project_name": f"{backend}-{args.run_id}-a{args.run_attempt}",
            "run_id": args.run_id,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "answerer_model": answerer,
            "answerer_provider": models.get("answerer_provider"),
            "judge_model": judge,
            "judge_provider": models.get("judge_provider"),
            "provider": models.get("answerer_provider"),
            "top_k": max(cutoffs),
            "top_k_cutoffs": labels_for(eval_cutoffs),
            "cutoffs": cutoffs,
            "total_questions": len(evaluations),
            group_key: groups,
            "memory_backend": {"mem0-oss": "oss", "none": "none"}.get(backend, "kg"),
            "backend": backend,
            "kg_variant": variant if backend.startswith("kg-") else None,
            "github_run_id": args.run_id,
            "run_attempt": args.run_attempt,
            "harness_sha": sha,
            "leg_harness_sha": leg_harness,
            "kg_commit": next((l["meta"].get("kg_commit") or (l["meta"].get("kg_meta") or {}).get("kg_commit") for l in legs if l["meta"].get("kg_commit") or (l["meta"].get("kg_meta") or {}).get("kg_commit")), None),
            "embedding_model": kg_meta.get("embedding_model") or kg_meta.get("kg_embed_provider") or first.get("embedding_model"),
            "agent_model": models.get("agent_model"),
            "agent_provider": models.get("agent_provider"),
            "mem0_llm_model": mem0.get("llm_model") or models.get("mem0_llm_model"),
            "mem0_search_flags": mem0.get("search_flags"),
            "mem0_image_digest": mem0.get("image_digest"),
            "legs": [leg_summary(l) for l in sorted(legs, key=lambda l: l["meta"].get("shard", 0))],
            "expected_shards": expected,
            "missing_shards": missing,
            "wall_seconds_total": sum(float(l["meta"].get("wall_seconds") or 0) for l in legs) or None,
            "answer_judge_calls": answer_judge_calls(evaluations),
            "merged_at": datetime.now().isoformat(timespec="seconds"),
        }
        metadata.update({k: v for k, v in kg_meta.items() if k not in metadata})

        save_result_json(args.out, {"metadata": metadata, "metrics_by_cutoff": metrics, "evaluations": evaluations})
        log(f"merge: {backend} {len(legs)} legs, {len(evaluations)} evaluations, shards {present}"
            + (f", MISSING {missing}" if missing else "") + f" -> {args.out}")
        for label, m in metrics.items():
            o = m.get("overall", {})
            log(f"merge:   {label}: {o.get('correct', 0)}/{o.get('total', 0)} ({o.get('accuracy', 0):.1f}%)")
        return 0
    except BenchError as exc:
        log(f"merge: {exc}")
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
