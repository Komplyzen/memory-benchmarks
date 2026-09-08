#!/usr/bin/env python3
"""
Strip dataset text from a merged result file before publishing (spec-benchmark-ci CAP-8)
=========================================================================================

    python -m scripts.strip_for_publish IN OUT [--keep-questions]

LoCoMo is CC BY-NC 4.0 and LongMemEval carries synthetic but licensed
conversation text; the published result files must not redistribute either.

Stripped from every evaluation (both benchmarks):
- ``retrieval.search_results[].memory``  (retrieved memory text; ids, scores,
  created_at and score_debug are kept)
- ``retrieval.search_query``             (equals the question text)
- ``question``                           (dataset question text)
- ``ground_truth_answer``                (dataset answer text)
- ``user_profile``                       (mem0 cloud profile, derived from dataset text)
- ``retrieval.query_debug.*`` is kept (counts only)
- ``evidence`` is kept for LoCoMo (dialog ids like "D1:3", not text)

Kept: ``question_id``, ``category``/``question_type``, ``cutoff_results[]`` with
``judgment``, ``score``, ``memories_evaluated``; ``generated_answer`` and the
judge ``reason``/``judge_raw`` are model output, not dataset text, and are kept.

``--keep-questions`` retains ``question``, ``ground_truth_answer`` and
``retrieval.search_query`` (for an internal, non-published copy).

The metadata block is copied unchanged, plus ``stripped_for_publish: true`` and
the list of stripped fields.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.common.utils import save_result_json

ALWAYS_STRIP = ["retrieval.search_results[].memory", "user_profile"]
QUESTION_FIELDS = ["question", "ground_truth_answer", "retrieval.search_query"]


def strip_evaluation(e: dict, keep_questions: bool) -> dict:
    out = dict(e)
    retrieval = dict(out.get("retrieval") or {})
    results = []
    for r in retrieval.get("search_results") or []:
        r2 = {k: v for k, v in r.items() if k != "memory"}
        results.append(r2)
    if "search_results" in retrieval:
        retrieval["search_results"] = results
    out.pop("user_profile", None)
    if not keep_questions:
        out.pop("question", None)
        out.pop("ground_truth_answer", None)
        retrieval.pop("search_query", None)
    if out.get("retrieval") is not None:
        out["retrieval"] = retrieval
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--keep-questions", action="store_true")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    evaluations = data.get("evaluations") or []
    stripped = [strip_evaluation(e, args.keep_questions) for e in evaluations]
    fields = list(ALWAYS_STRIP) + ([] if args.keep_questions else QUESTION_FIELDS)
    metadata = dict(data.get("metadata") or {})
    metadata["stripped_for_publish"] = True
    metadata["stripped_fields"] = fields

    save_result_json(args.output, {"metadata": metadata, "metrics_by_cutoff": data.get("metrics_by_cutoff", {}), "evaluations": stripped})
    print(f"strip: {len(stripped)} evaluations, removed {fields} -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
