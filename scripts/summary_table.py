#!/usr/bin/env python3
"""
Job-summary markdown from merged result files (spec-benchmark-ci CAP-7)
======================================================================

    python -m scripts.summary_table --out SUMMARY.md [--full] [--failed backend:shard:phase ...] MERGED.json...

Header: run id, harness SHA, KG commit, models, cutoffs, "all legs from this
run", "each backend embeds with its own provider", dev-loop notice with the
shards covered when not --full.

One row per backend: category (or question-type) accuracy per cutoff, overall
per cutoff, mean tokens to answerer per cutoff, p50/p95 search latency, nodes
ingested per shard, embed failures, wall time, actual LLM calls (kg ingest
calls from metadata + answer/judge calls counted from evaluations), embedding
model; for mem0-oss its search flags and LLM_MODEL. Delta rows (percentage
points) for each kg-* backend vs mem0-oss when present. One line per --failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from benchmarks.common.bench_common import KG_BACKENDS, answer_judge_calls, latency_percentiles, tokens_to_answerer
from benchmarks.common.utils import FULL_CONTEXT_CUTOFF, cutoff_label


def fmt(v: Any, nd: int = 1, suffix: str = "") -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def fmt_secs(s: float | None) -> str:
    if not s:
        return "–"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def groups_of(data: dict[str, Any]) -> tuple[str, list[str]]:
    mbc = data.get("metrics_by_cutoff") or {}
    for m in mbc.values():
        if "by_category" in m:
            return "by_category", sorted(m["by_category"])
        if "by_question_type" in m:
            return "by_question_type", sorted(m["by_question_type"])
    return "by_category", []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--full", action="store_true", help="omit the dev-loop notice")
    p.add_argument("--failed", action="append", default=[], help="backend:shard:phase for each failed leg")
    p.add_argument("--title", default="Public memory benchmark")
    p.add_argument("files", nargs="*")
    args = p.parse_args()

    runs = [load(f) for f in args.files]
    by_backend: dict[str, dict[str, Any]] = {}
    for r in runs:
        b = (r.get("metadata") or {}).get("backend") or (r.get("metadata") or {}).get("memory_backend") or "?"
        by_backend[b] = r

    lines: list[str] = [f"## {args.title}", ""]

    if runs:
        md0 = runs[0]["metadata"]
        lines += [
            f"- Run id: `{md0.get('github_run_id')}` attempt {md0.get('run_attempt')}",
            f"- Harness SHA: `{md0.get('harness_sha')}`",
            f"- KG commit: `{md0.get('kg_commit')}`",
            f"- Models: answerer `{md0.get('answerer_model')}` ({md0.get('answerer_provider')}), judge `{md0.get('judge_model')}` ({md0.get('judge_provider')}), agent `{md0.get('agent_model')}` ({md0.get('agent_provider')}), mem0 extraction `{md0.get('mem0_llm_model')}`",
            f"- Cutoffs: {', '.join(str(c) for c in (md0.get('cutoffs') or []))}"
            + (" (none backend: full_context)" if any("full_context" in (r.get("metrics_by_cutoff") or {}) for r in runs) else ""),
            "- All legs come from this run; no numbers are mixed across runs.",
            "- Each backend embeds with its own provider (KG: its configured embedding provider; mem0: its container config).",
        ]
        if not args.full:
            shards = sorted({leg.get("shard") for r in runs for leg in (r["metadata"].get("legs") or []) if leg.get("shard") is not None})
            lines.append(f"- **Dev-loop sample, not a full run.** Shards covered: {shards}. Do not quote as a benchmark result.")
        missing = {b: r["metadata"].get("missing_shards") for b, r in by_backend.items() if r["metadata"].get("missing_shards")}
        if missing:
            lines.append(f"- **Partial:** missing shards {missing}")
        lines.append("")

    # ---- main table -------------------------------------------------------
    if runs:
        group_field, groups = groups_of(runs[0])
        for r in runs[1:]:
            _, g = groups_of(r)
            groups = sorted(set(groups) | set(g))
        cutoffs = runs[0]["metadata"].get("cutoffs") or []
        labels = [cutoff_label(c) for c in cutoffs]
        # full_context columns only when a none (no-memory) backend actually ran
        has_full_context = any("full_context" in (r.get("metrics_by_cutoff") or {}) for r in runs)
        all_labels = labels + (["full_context"] if has_full_context else [])
        token_cutoffs = list(zip(labels, cutoffs)) + ([("full_context", FULL_CONTEXT_CUTOFF)] if has_full_context else [])
        cols = ["backend"]
        for lab in all_labels:
            cols += [f"{g}@{lab}" for g in groups] + [f"overall@{lab}"]
        cols += [f"tok→ans@{lab}" for lab in all_labels]
        cols += ["p50 ms", "p95 ms", "nodes/shard", "embed fail", "wall", "LLM calls", "embedding", "notes"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))

        def row_for(b: str, r: dict[str, Any]) -> list[str]:
            md, mbc, ev = r["metadata"], r.get("metrics_by_cutoff") or {}, r.get("evaluations") or []
            cells = [f"**{b}**"]
            for lab in all_labels:
                m = mbc.get(lab) or {}
                grp = m.get(group_field) or {}
                cells += [fmt((grp.get(g) or {}).get("accuracy")) for g in groups]
                cells.append(fmt((m.get("overall") or {}).get("accuracy")))
            for lab, c in token_cutoffs:
                cells.append(fmt(tokens_to_answerer(ev, c), 0) if lab in mbc else "–")
            p50, p95 = latency_percentiles(ev)
            cells += [fmt(p50), fmt(p95)]
            legs = md.get("legs") or []
            nodes = [l.get("nodes_created") for l in legs if l.get("nodes_created") is not None]
            cells.append("/".join(str(n) for n in nodes) if nodes else "–")
            ef = [l.get("embed_failures") for l in legs if l.get("embed_failures") is not None]
            cells.append(str(sum(ef)) if ef else "–")
            cells.append(fmt_secs(md.get("wall_seconds_total")))
            ingest_calls = md.get("kg_ingest_llm_calls") or 0
            cells.append(f"{ingest_calls + answer_judge_calls(ev)} ({ingest_calls} ingest)" if b in KG_BACKENDS else str(answer_judge_calls(ev)))
            cells.append(str(md.get("embedding_model") or "–"))
            notes = []
            if b == "mem0-oss":
                notes.append(f"LLM_MODEL={md.get('mem0_llm_model')}, search={json.dumps(md.get('mem0_search_flags'))}")
            if b in KG_BACKENDS:
                notes.append(f"variant={md.get('kg_variant')}, cache hit {fmt((md.get('kg_ingest_cache_hit_ratio') or 0) * 100, 0, '%')}, max_tool_calls hit {md.get('kg_ingest_max_tool_calls_hit', 0)}×")
            if b == "none":
                notes.append(f"truncated {md.get('none_questions_truncated', 0)} q, dropped {md.get('none_sessions_dropped_total', 0)} sessions")
            cells.append("; ".join(notes) or "–")
            return cells

        for b in sorted(by_backend):
            lines.append("| " + " | ".join(row_for(b, by_backend[b])) + " |")

        # ---- delta rows vs mem0-oss ------------------------------------------
        base = by_backend.get("mem0-oss")
        if base:
            bm = base.get("metrics_by_cutoff") or {}
            for b in sorted(by_backend):
                if b not in KG_BACKENDS:
                    continue
                km = by_backend[b].get("metrics_by_cutoff") or {}
                cells = [f"Δ {b} vs mem0-oss (pp)"]
                for lab in all_labels:
                    for g in groups + ["overall"]:
                        if g == "overall":
                            a, o = (km.get(lab) or {}).get("overall", {}).get("accuracy"), (bm.get(lab) or {}).get("overall", {}).get("accuracy")
                        else:
                            a = ((km.get(lab) or {}).get(group_field) or {}).get(g, {}).get("accuracy")
                            o = ((bm.get(lab) or {}).get(group_field) or {}).get(g, {}).get("accuracy")
                        cells.append(f"{a - o:+.1f}" if a is not None and o is not None else "–")
                cells += ["–"] * (len(cols) - len(cells))
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # ---- failed legs -------------------------------------------------------
    if args.failed:
        lines.append("### Failed legs")
        for entry in args.failed:
            parts = entry.split(":")
            backend = parts[0] if parts else "?"
            shard = parts[1] if len(parts) > 1 else "?"
            phase = parts[2] if len(parts) > 2 else "unknown"
            lines.append(f"- `{backend}` shard {shard}: failed in **{phase}**")
        lines.append("")

    if not runs and not args.failed:
        lines.append("_No merged results._")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"summary: {len(runs)} backends, {len(args.failed)} failed legs -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
