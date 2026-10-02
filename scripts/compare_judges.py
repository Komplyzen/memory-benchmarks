"""Re-judge saved LoCoMo answers with Jev and compare against the original judge.

Reads ``predicted_<project>/<qid>.json`` files written by benchmarks/locomo/run.py. For every
cutoff it takes the STORED ``generated_answer`` (nothing is regenerated, unlike the harness's
--rejudge, which re-runs the answerer), asks Jev for a verdict, and compares it with the stored
``judgment``. Spends only Jev tokens. Never writes to the result directory.

  python -m scripts.compare_judges <predicted_dir> [--out DIR] [--threshold 0.5] [--limit N]
                                   [--concurrency 8] [--dry-run]

--dry-run builds and prints the first request and exits; needs no API key and makes no call.
Limitation: the original run must not have used --with-evidence (this tool does not pass evidence).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from benchmarks.common.jev_judge import JevError, JevJudge, JevVerdict, build_request

UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.2, 0.8
Judge = Callable[[int, str, str, str], Awaitable[JevVerdict]]


def load_items(predicted_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """One item per (question, cutoff) that has a stored judgment and generated answer."""
    items: list[dict[str, Any]] = []
    for path in sorted(predicted_dir.glob("conv*_q*.json")):
        data = json.loads(path.read_text())
        for label, cut in (data.get("cutoff_results") or {}).items():
            if "judgment" not in cut or "generated_answer" not in cut:
                continue
            items.append({
                "question_id": data["question_id"], "category": data["category"],
                "category_name": data.get("category_name", ""), "cutoff": label,
                "question": data["question"], "gold": data["ground_truth_answer"],
                "generated": cut["generated_answer"],
                "original_correct": cut["judgment"] == "CORRECT",
                "original_reason": cut.get("reason", ""),
            })
    return items[:limit] if limit else items


async def judge_all(items: list[dict[str, Any]], judge: Judge) -> list[dict[str, Any]]:
    """Attach ``jev_p_yes``/``jev_correct`` to each item; failures get ``error`` and no verdict."""
    async def one(item: dict[str, Any]) -> dict[str, Any]:
        try:
            v = await judge(item["category"], item["question"], item["gold"], item["generated"])
        except JevError as exc:
            return {**item, "error": str(exc)}
        return {**item, "jev_p_yes": v.p_yes, "jev_correct": v.correct,
                "input_tokens": v.input_tokens, "output_tokens": v.output_tokens}
    return list(await asyncio.gather(*(one(i) for i in items)))


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement statistics over the rows that got a verdict. Pure."""
    ok = [r for r in rows if "jev_correct" in r]
    orig = [r["original_correct"] for r in ok]
    jev = [r["jev_correct"] for r in ok]
    agree = [r for r in ok if r["original_correct"] == r["jev_correct"]]

    def rate(sub: list[dict[str, Any]]) -> float | None:
        return sum(1 for r in sub if r["original_correct"] == r["jev_correct"]) / len(sub) if sub else None

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cut: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        by_cat[f'{r["category"]} {r["category_name"]}'.strip()].append(r)
        by_cut[r["cutoff"]].append(r)
    confusion = Counter((r["original_correct"], r["jev_correct"]) for r in ok)
    return {
        "judged": len(ok), "errors": len(rows) - len(ok),
        "agreement": rate(ok), "kappa": cohens_kappa(orig, jev),
        "original_accuracy": sum(orig) / len(ok) if ok else None,
        "jev_accuracy": sum(jev) / len(ok) if ok else None,
        "confusion": {"both_correct": confusion[(True, True)], "both_wrong": confusion[(False, False)],
                      "original_correct_jev_wrong": confusion[(True, False)],
                      "original_wrong_jev_correct": confusion[(False, True)]},
        "uncertain_jev": sum(1 for r in ok if UNCERTAIN_LOW < r["jev_p_yes"] < UNCERTAIN_HIGH),
        "by_category": {k: {"n": len(v), "agreement": rate(v)} for k, v in sorted(by_cat.items())},
        "by_cutoff": {k: {"n": len(v), "agreement": rate(v)} for k, v in sorted(by_cut.items())},
        "disagreements": [r for r in ok if r["original_correct"] != r["jev_correct"]],
        "tokens": {"input": sum(r["input_tokens"] for r in ok), "output": sum(r["output_tokens"] for r in ok)},
        "agreed": len(agree),
    }


def render_markdown(s: dict[str, Any], threshold: float, source: str) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"
    k = "n/a" if s["kappa"] is None else f'{s["kappa"]:.2f}'
    c = s["confusion"]
    lines = [
        "# Judge comparison: original judge vs Jev", "",
        f"Source: `{source}`  ·  Jev threshold: p(yes) > {threshold}", "",
        f'- Judged **{s["judged"]}** answers ({s["errors"]} Jev errors, excluded)',
        f'- Agreement **{pct(s["agreement"])}**, Cohen\'s kappa **{k}**',
        f'- Accuracy: original **{pct(s["original_accuracy"])}**, Jev **{pct(s["jev_accuracy"])}**',
        f'- Jev uncertain ({UNCERTAIN_LOW} < p < {UNCERTAIN_HIGH}): **{s["uncertain_jev"]}**',
        f'- Jev tokens: {s["tokens"]["input"]} in / {s["tokens"]["output"]} out', "",
        "| | Jev correct | Jev wrong |", "|---|---|---|",
        f'| original correct | {c["both_correct"]} | {c["original_correct_jev_wrong"]} |',
        f'| original wrong | {c["original_wrong_jev_correct"]} | {c["both_wrong"]} |', "",
        "## By category", "", "| category | n | agreement |", "|---|---|---|",
        *[f'| {k} | {v["n"]} | {pct(v["agreement"])} |' for k, v in s["by_category"].items()], "",
        "## By cutoff", "", "| cutoff | n | agreement |", "|---|---|---|",
        *[f'| {k} | {v["n"]} | {pct(v["agreement"])} |' for k, v in s["by_cutoff"].items()], "",
        f'## Disagreements ({len(s["disagreements"])})', "",
    ]
    for d in s["disagreements"]:
        orig = "CORRECT" if d["original_correct"] else "WRONG"
        lines += [f'### {d["question_id"]} · {d["cutoff"]} · original {orig}, Jev p={d["jev_p_yes"]:.2f}', "",
                  f'- Q: {d["question"]}', f'- Gold: {d["gold"]}', f'- Generated: {d["generated"]}',
                  f'- Original reason: {d["original_reason"]}', ""]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    items = load_items(Path(args.predicted_dir), args.limit)
    if not items:
        print(f"no judged answers found under {args.predicted_dir}", file=sys.stderr)
        return 2
    if args.dry_run:
        first = items[0]
        print(f"{len(items)} answers to judge. First request:")
        print(json.dumps(build_request(first["category"], first["question"], first["gold"], first["generated"]), indent=2))
        return 0
    try:
        jev_ctx = JevJudge(threshold=args.threshold, concurrency=args.concurrency)
    except JevError as exc:  # no API key: say so plainly instead of a traceback
        print(f"compare_judges: {exc}", file=sys.stderr)
        return 2
    async with jev_ctx as jev:
        rows = await judge_all(items, jev.judge)
    summary = summarize(rows)
    out = Path(args.out or args.predicted_dir) / f"judge_compare_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    out.with_suffix(".md").write_text(render_markdown(summary, args.threshold, str(args.predicted_dir)))
    print(f"wrote {out}.md and .json  (agreement {summary['agreement']}, kappa {summary['kappa']})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predicted_dir")
    p.add_argument("--out", default=None, help="report directory (default: next to the results)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="judge only the first N answers")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
