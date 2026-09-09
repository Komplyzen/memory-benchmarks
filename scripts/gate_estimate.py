#!/usr/bin/env python3
"""
Gate + estimate for bench.sh / benchmark.yml (spec-benchmark-ci CAP-1)
=====================================================================

Validates the run inputs, downloads and verifies the dataset, resolves shards
and models, estimates LLM calls and EUR, and prints ONE JSON object on stdout.
Human-readable lines go to stderr.

Exit codes:
  0  ok
  2  invalid input (unknown backend, bad cutoff, bad shard, unknown profile)
  3  dataset sha256 mismatch (both hashes printed)
  4  estimate above the ceiling (2,500 LLM calls) without --full
  5  KG_EMBED_FIXTURES is set (benchmarks must use the real embedding provider)

Estimate model (per backend):
  calls   = questions * len(cutoffs) * 2            (answer + judge)
          + sessions * 4   for kg-*                  (agent ingest)
          + sessions * 1   for mem0-oss              (extraction)
          + 0              for none                  (no ingest; one pseudo-cutoff)
  tokens  agent call   ~7,000 in (4,000 of them cacheable) / 300 out
          answerer     ~60 per memory at the cutoff + 400 in / 100 out
                       (none: full transcript tokens + 400 in)
          judge        ~600 in / 50 out
  eur     from benchmarks/common/profiles.json list prices (dated), USD->EUR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from typing import Any

from benchmarks.common.bench_common import (
    KG_BACKENDS,
    BenchError,
    MODEL_KEYS,
    approx_tokens,
    load_profiles,
    normalize_backends,
    normalize_cutoffs,
    required_env,
    resolve_endpoints,
    resolve_models,
    usd_cost,
)

# The spend guard is a budget in EUR (--budget-eur), not a call ceiling: exit 4 when the forecast
# is above it. bench.sh then asks a person (or honours --yes). Nothing is spent by this script.
DEFAULT_BUDGET_EUR = 5.0

AGENT_CALLS_PER_SESSION = 4
AGENT_IN, AGENT_CACHEABLE, AGENT_OUT = 7000, 4000, 300
MEM0_EXTRACT_CALLS_PER_SESSION = 1
MEM0_EXTRACT_IN, MEM0_EXTRACT_OUT = 2500, 300
ANSWERER_PER_MEMORY, ANSWERER_BASE_IN, ANSWERER_OUT = 60, 400, 100
JUDGE_IN, JUDGE_OUT = 600, 50


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataset access via the runners' own functions
# ---------------------------------------------------------------------------


def resolve_dataset(benchmark: str, dataset_path: str | None) -> str:
    if dataset_path:
        return dataset_path
    logger = logging.getLogger("gate")
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    if benchmark == "locomo":
        from benchmarks.locomo.run import DEFAULT_DATASET_DIR, download_dataset
    else:
        from benchmarks.longmemeval.run import DEFAULT_DATASET_DIR, download_dataset
    return download_dataset(DEFAULT_DATASET_DIR, logger)


def locomo_shards(data: list[dict], shards_arg: str, max_questions: int | None) -> list[dict[str, Any]]:
    from benchmarks.locomo.prompts import CATEGORIES_TO_EVALUATE
    from benchmarks.locomo.run import get_sorted_sessions

    indices = list(range(len(data))) if shards_arg.strip() == "all" else parse_indices(shards_arg)
    out = []
    for idx in indices:
        if idx < 0 or idx >= len(data):
            raise BenchError(f"shard {idx} out of range for locomo ({len(data)} conversations)", 2)
        entry = data[idx]
        qa = entry.get("qa", entry.get("qa_pairs", []))
        questions = [q for q in qa if q.get("category") in CATEGORIES_TO_EVALUATE]
        if max_questions is not None:
            questions = questions[:max_questions]
        sessions = get_sorted_sessions(entry["conversation"])
        transcript_chars = sum(len(t.get("text", "")) for _, _, turns in sessions for t in turns)
        out.append({
            "idx": idx,
            "questions": len(questions),
            "sessions": len(sessions),
            "transcript_tokens": approx_tokens(" " * transcript_chars),
        })
    return out


def longmemeval_shards(data: list[dict], shards_arg: str, shard_size: int, max_questions: int | None) -> list[dict[str, Any]]:
    n_shards = math.ceil(len(data) / shard_size)
    indices = list(range(n_shards)) if shards_arg.strip() == "all" else parse_indices(shards_arg)
    out = []
    for idx in indices:
        if idx < 0 or idx >= n_shards:
            raise BenchError(f"shard {idx} out of range for longmemeval ({n_shards} shards of {shard_size})", 2)
        questions = data[idx * shard_size:(idx + 1) * shard_size]
        if max_questions is not None:
            questions = questions[:max_questions]
        sessions = sum(len(q.get("haystack_sessions", [])) for q in questions)
        transcript_chars = sum(
            len(t.get("content", "")) for q in questions for s in q.get("haystack_sessions", []) for t in s
        )
        out.append({
            "idx": idx,
            "questions": len(questions),
            "sessions": sessions,
            "transcript_tokens": approx_tokens(" " * transcript_chars) // max(1, len(questions)),
            "dataset_path": f"longmemeval_s_shard{idx}.json",
        })
    return out


def parse_indices(csv: str) -> list[int]:
    try:
        vals = [int(x.strip()) for x in csv.split(",") if x.strip()]
    except ValueError as exc:
        raise BenchError(f"shards must be a comma list of integers or 'all': {csv!r}", 2) from exc
    if not vals:
        raise BenchError("no shards given", 2)
    return sorted(set(vals))


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


def estimate_backend(backend: str, shards: list[dict[str, Any]], cutoffs: list[int], models: dict[str, str],
                     profiles: dict[str, Any], benchmark: str) -> dict[str, Any]:
    questions = sum(s["questions"] for s in shards)
    sessions = sum(s["sessions"] for s in shards)
    n_cutoffs = 1 if backend == "none" else len(cutoffs)

    usd = 0.0
    missing: set[str] = set()

    def add(model: str, in_t: float, out_t: float, cached: float = 0.0) -> None:
        nonlocal usd
        c = usd_cost(model, profiles, in_t, out_t, cached)
        if c is None:
            missing.add(model)
        else:
            usd += c

    # ingest
    if backend in KG_BACKENDS:
        ingest_calls = sessions * AGENT_CALLS_PER_SESSION
        add(models["agent_model"], ingest_calls * (AGENT_IN - AGENT_CACHEABLE), ingest_calls * AGENT_OUT, ingest_calls * AGENT_CACHEABLE)
    elif backend == "mem0-oss":
        ingest_calls = sessions * MEM0_EXTRACT_CALLS_PER_SESSION
        add(models["mem0_llm_model"], ingest_calls * MEM0_EXTRACT_IN, ingest_calls * MEM0_EXTRACT_OUT)
    else:
        ingest_calls = 0

    # answer + judge
    qa_calls = questions * n_cutoffs * 2
    if backend == "none":
        # one answer per question with the full transcript in context
        if benchmark == "locomo":
            transcript_in = sum(s["questions"] * s["transcript_tokens"] for s in shards)
        else:
            transcript_in = sum(s["questions"] * s["transcript_tokens"] for s in shards)
        add(models["answerer_model"], transcript_in + questions * ANSWERER_BASE_IN, questions * ANSWERER_OUT)
        add(models["judge_model"], questions * JUDGE_IN, questions * JUDGE_OUT)
    else:
        for c in cutoffs:
            add(models["answerer_model"], questions * (c * ANSWERER_PER_MEMORY + ANSWERER_BASE_IN), questions * ANSWERER_OUT)
            add(models["judge_model"], questions * JUDGE_IN, questions * JUDGE_OUT)

    eur = round(usd * profiles.get("usd_to_eur", 1.0), 2)
    return {"calls": ingest_calls + qa_calls, "ingest_calls": ingest_calls, "qa_calls": qa_calls,
            "eur": eur, "price_missing": sorted(missing)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate inputs, verify dataset, estimate calls and EUR; JSON on stdout.")
    p.add_argument("--benchmark", required=True, choices=["locomo", "longmemeval"])
    p.add_argument("--shards", required=True, help="comma list of shard indices or 'all'")
    p.add_argument("--shard-size", type=int, default=25, help="longmemeval questions per shard")
    p.add_argument("--cutoffs", required=True, help="comma list, normalized (sorted, deduped)")
    p.add_argument("--backends", required=True, help="comma list of kg-full,kg-no-spread,kg-no-decay,mem0-oss,none")
    p.add_argument("--profile", required=True, help="profile name from benchmarks/common/profiles.json")
    for k in MODEL_KEYS:
        p.add_argument(f"--{k.replace('_', '-')}", default=None, help=f"override {k} from the profile")
    p.add_argument("--dataset-path", default=None, help="local dataset file; downloaded with the runner's function if omitted")
    p.add_argument("--expected-sha256", default=None, help="exit 3 if the dataset file's sha256 differs")
    p.add_argument("--max-questions", type=int, default=None, help="cap questions per shard in the estimate")
    p.add_argument("--budget-eur", type=float, default=DEFAULT_BUDGET_EUR,
                   help="spend limit for this run in EUR; forecast above it -> exit 4 (bench.sh asks for approval)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if os.getenv("KG_EMBED_FIXTURES"):
            raise BenchError("KG_EMBED_FIXTURES is set; benchmarks must embed with the real provider", 5)

        profiles = load_profiles()
        backends = normalize_backends(args.backends)
        models = resolve_models(args.profile, {k: getattr(args, k) for k in MODEL_KEYS}, profiles)
        # Profile-mandated extra legs (e.g. small-model adds the no-memory floor).
        for extra in profiles["profiles"][args.profile].get("extra_backends") or []:
            name = normalize_backends(extra)[0]
            if name not in backends:
                backends.append(name)
                log(f"profile {args.profile} adds backend {name}")
        cutoffs = normalize_cutoffs(args.cutoffs, backends)

        dataset_path = resolve_dataset(args.benchmark, args.dataset_path)
        digest = sha256_file(dataset_path)
        if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
            log(f"dataset sha256 mismatch\n  expected: {args.expected_sha256.lower()}\n  actual:   {digest}")
            return 3
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        if args.benchmark == "locomo":
            shards = locomo_shards(data, args.shards, args.max_questions)
            for s in shards:
                s["dataset_path"] = dataset_path
            all_shards = len(data)
        else:
            shards = longmemeval_shards(data, args.shards, args.shard_size, args.max_questions)
            all_shards = math.ceil(len(data) / args.shard_size)
        # "complete" = every shard of the dataset, no question cap: a fact, not a flag.
        complete = len(shards) == all_shards and not args.max_questions

        per_backend = {b: estimate_backend(b, shards, cutoffs, models, profiles, args.benchmark) for b in backends}
        total_calls = sum(v["calls"] for v in per_backend.values())
        total_eur = round(sum(v["eur"] for v in per_backend.values()), 2)
        price_missing = sorted({m for v in per_backend.values() for m in v["price_missing"]})

        endpoints = resolve_endpoints(args.profile, {k: getattr(args, k) for k in MODEL_KEYS}, profiles)
        env_needed = required_env(endpoints)
        out = {
            "benchmark": args.benchmark,
            "profile": args.profile,
            "models": models,
            "endpoints": endpoints,
            "required_env": env_needed,
            "cutoffs": cutoffs,
            "backends": backends,
            "shards": [{k: v for k, v in s.items() if k != "transcript_tokens"} for s in shards],
            "totals": {
                "questions": sum(s["questions"] for s in shards),
                "sessions": sum(s["sessions"] for s in shards),
                "calls": total_calls,
                "eur": total_eur,
            },
            "per_backend": {b: {"calls": v["calls"], "eur": v["eur"]} for b, v in per_backend.items()},
            "dataset_sha256": digest,
            "prices_recorded_on": profiles.get("prices_recorded_on"),
            "price_missing": price_missing,
            "budget_eur": args.budget_eur,
            "over_budget": total_eur > args.budget_eur,
            "complete": complete,
        }

        log(f"gate: {args.benchmark} profile={args.profile} shards={[s['idx'] for s in shards]} backends={backends} cutoffs={cutoffs}")
        log(f"gate: models answerer={models['answerer_model']}/{models['answerer_provider']} judge={models['judge_model']}/{models['judge_provider']} "
            f"agent={models['agent_model']}/{models['agent_provider']} mem0_llm={models['mem0_llm_model']}")
        log(f"gate: required env for these endpoints: {', '.join(env_needed)}"
            + (f"  (unset here: {', '.join(n for n in env_needed if not os.getenv(n))})" if any(not os.getenv(n) for n in env_needed) else ""))
        for b, v in per_backend.items():
            log(f"gate:   {b:<13} calls={v['calls']:>6} (ingest {v['ingest_calls']}, answer+judge {v['qa_calls']})  ~EUR {v['eur']:.2f}")
        log(f"gate: total calls={total_calls} ~EUR {total_eur:.2f} (list prices recorded {profiles.get('prices_recorded_on')}, verify) dataset sha256={digest[:16]}...")
        if price_missing:
            log(f"gate: WARNING no price for {price_missing}; their cost is not included")

        print(json.dumps(out))

        if total_eur > args.budget_eur:
            log(f"gate: forecast EUR {total_eur:.2f} is above the budget of EUR {args.budget_eur:.2f} ({total_calls} calls); approval needed")
            return 4
        return 0
    except BenchError as exc:
        log(f"gate: {exc}")
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
