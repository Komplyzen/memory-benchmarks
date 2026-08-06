"""Lift metrics out of a benchmark's result file into the ledger.

Per-benchmark extractors normalize to a compact dict:
  {"headline": <float|None>, "headline_label": str,
   "by_cutoff": {label: accuracy}, "total_questions": int}
The full detail stays in the result file; the ledger holds what you need for
`ls` and `diff` -- what did we get, at a glance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _acc(block: dict[str, Any]) -> Optional[float]:
    """Accuracy % from a metrics block, tolerating shape variants."""
    if not isinstance(block, dict):
        return None
    overall = block.get("overall") if isinstance(block.get("overall"), dict) else block
    acc = overall.get("accuracy")
    if acc is None and overall.get("total"):
        try:
            acc = overall["correct"] / overall["total"] * 100
        except Exception:
            acc = None
    return round(acc, 2) if isinstance(acc, (int, float)) else None


def extract_longmemeval(result_file: str) -> Optional[dict[str, Any]]:
    p = Path(result_file)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    mbc = data.get("metrics_by_cutoff") or {}
    by_cutoff = {}
    for label, block in mbc.items():
        a = _acc(block)
        if a is not None:
            by_cutoff[label] = a
    meta = data.get("metadata") or {}
    # Integrity: empty generated answers = exhausted LLM retries silently scored
    # wrong. Count them so a poisoned run is visible, never silent.
    empty_answers = 0
    for e in data.get("evaluations") or []:
        cr = e.get("cutoff_results") or {}
        if cr and any(not (v.get("generated_answer") or "").strip() for v in cr.values()):
            empty_answers += 1
    headline_label = None
    headline = None
    if by_cutoff:
        # highest cutoff = the run's configured top_k ceiling; use as headline
        def _k(label: str) -> int:
            digits = "".join(ch for ch in label if ch.isdigit())
            return int(digits) if digits else 0
        headline_label = max(by_cutoff, key=_k)
        headline = by_cutoff[headline_label]
    # Producer-named verbatim pairs, ordered: what UIs print, exactly as named here.
    def _cut(label):
        d = "".join(ch for ch in label if ch.isdigit())
        return int(d) if d else 0
    # Highest cutoff first: the headline (first pair) is acc@<max cutoff>.
    pairs = {f"acc@{_cut(label) or label}": v
             for label, v in sorted(by_cutoff.items(), key=lambda kv: -_cut(kv[0]))}
    out = {
        "pairs": pairs,
        "headline": headline,
        "headline_label": headline_label,
        "by_cutoff": by_cutoff,
        "total_questions": meta.get("total_questions"),
        "mode": meta.get("mode"),
    }
    if empty_answers:
        out["INTEGRITY_WARNING"] = (
            f"{empty_answers} questions have EMPTY generated answers "
            "(LLM retries exhausted, scored wrong) -- metrics are biased low; rerun those."
        )
        out["empty_answers"] = empty_answers
    return out


def extract_browsecomp_plus(result_file: str) -> Optional[dict[str, Any]]:
    path = Path(result_file)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    evidence = (data.get("metrics") or {}).get("evidence") or {}
    pairs = {key: value for key, value in evidence.items() if isinstance(value, (int, float))}
    latency = data.get("latency_ms") or {}
    for key in ("p50", "p95", "mean"):
        if isinstance(latency.get(key), (int, float)):
            pairs[f"latency_{key}_ms"] = latency[key]
    headline_label = next((key for key in evidence if key.startswith("ndcg_at_")), None)
    return {
        "pairs": pairs,
        "headline": evidence.get(headline_label) if headline_label else None,
        "headline_label": headline_label,
        "total_questions": data.get("query_count"),
        "mode": data.get("search_mode"),
        "failure_count": data.get("failure_count", 0),
        "canonical": data.get("canonical"),
    }


_EXTRACTORS = {
    "longmemeval": extract_longmemeval,
    "browsecomp_plus": extract_browsecomp_plus,
}


def extract(benchmark: str, result_file: str) -> Optional[dict[str, Any]]:
    fn = _EXTRACTORS.get(benchmark)
    return fn(result_file) if fn else None
