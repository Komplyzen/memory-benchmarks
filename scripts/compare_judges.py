"""Re-judge saved LoCoMo answers with Jev and compare against the original judge.

Reads ``predicted_<project>/<qid>.json`` files written by benchmarks/locomo/run.py. For every
cutoff it takes the STORED ``generated_answer`` (nothing is regenerated, unlike the harness's
--rejudge, which re-runs the answerer), asks Jev for a verdict, and compares it with the stored
``judgment``. Spends only Jev tokens. Never writes into the result directory: reports go to
``--out`` (default: ``judge_compare/`` next to it).

Resumable: every verdict is appended to ``jev_rows_<run id>.jsonl`` as it arrives. The run id is a
hash of the source directory, the Jev criteria and the model, so rerunning the same comparison
reuses finished verdicts (matched by question, cutoff and an input hash) and only calls Jev for
rows that are missing or errored. Only ``p(yes)`` is stored; the threshold is applied at report
time, so changing ``--threshold`` never needs new calls.

Answers whose original judge call failed are excluded from the comparison and counted separately.
run.py records ``judge_failed`` for this; result files written before that field existed fall back
to "WRONG with an empty reason" and the report says how many were inferred that way.

  python -m scripts.compare_judges <predicted_dir> [--out DIR] [--threshold 0.5] [--limit N]
                                   [--seed 0] [--concurrency 8] [--breaker 10] [--dry-run]

--limit N (a positive integer) judges a deterministic sample of N answers, drawn after excluding
failed originals: seeded shuffle within each (category, cutoff) group, then round-robin across
groups, so the sample spreads over conversations, categories and cutoffs.

--dry-run builds and prints the first request and exits; needs no API key and makes no call.
Limitation: the original run must not have used --with-evidence (this tool does not pass evidence).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from benchmarks.common.jev_judge import (
    JEV_CRITERIA_FALSE, JEV_CRITERIA_TRUE, JEV_INSTRUCTIONS, JEV_MODEL, JevError, JevJudge, JevVerdict,
    build_request,
)

UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.2, 0.8
Judge = Callable[[int, str, str, str], Awaitable[JevVerdict]]


def _spread(items: list[dict[str, Any]], limit: int, seed: int = 0) -> list[dict[str, Any]]:
    """``limit`` items: seeded shuffle inside each (category, cutoff) group, then round-robin over groups."""
    rng = random.Random(seed)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        groups[(it["category"], it["cutoff"])].append(it)
    queues = []
    for k in sorted(groups):
        q = list(groups[k])
        rng.shuffle(q)
        queues.append(q)
    picked: list[dict[str, Any]] = []
    for rank in range(max(map(len, queues), default=0)):
        for q in queues:
            if rank < len(q) and len(picked) < limit:
                picked.append(q[rank])
    return picked


def load_items(predicted_dir: Path) -> list[dict[str, Any]]:
    """One item per (question, cutoff) that has a stored judgment and generated answer."""
    items: list[dict[str, Any]] = []
    for path in sorted(predicted_dir.glob("conv*_q*.json")):
        data = json.loads(path.read_text())
        for label, cut in (data.get("cutoff_results") or {}).items():
            if "judgment" not in cut or "generated_answer" not in cut:
                continue
            flagged = "judge_failed" in cut  # written by run.py; absent in older result files
            items.append({
                "question_id": data["question_id"], "category": data["category"],
                "category_name": data.get("category_name", ""), "cutoff": label,
                "question": data["question"], "gold": data["ground_truth_answer"],
                "generated": cut["generated_answer"],
                "original_correct": cut["judgment"] == "CORRECT",
                "original_reason": cut.get("reason", ""),
                "original_judge_failed": bool(cut["judge_failed"]) if flagged else not cut.get("reason", ""),
                "original_failure_inferred": not flagged and not cut.get("reason", ""),
            })
    return items


def select_items(items: list[dict[str, Any]], limit: int | None, seed: int = 0) -> list[dict[str, Any]]:
    """Failed originals first (they are never sent), then the sample is drawn from the rest only."""
    failed = [i for i in items if i["original_judge_failed"]]
    eligible = [i for i in items if not i["original_judge_failed"]]
    return (_spread(eligible, limit, seed) if limit else eligible) + failed


def input_hash(item: dict[str, Any]) -> str:
    """Identity of what Jev is asked: a changed question, gold answer or generated answer re-judges."""
    blob = json.dumps([item["question"], str(item["gold"]), item["generated"]], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def criteria_hash() -> str:
    blob = json.dumps([JEV_MODEL, JEV_INSTRUCTIONS, JEV_CRITERIA_TRUE, JEV_CRITERIA_FALSE])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_id(predicted_dir: Path) -> str:
    """Stable identity of one comparison: same source, same criteria, same model."""
    blob = f"{predicted_dir.resolve()}|{criteria_hash()}"
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["question_id"], row["cutoff"]


def load_progress(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Finished verdicts from a previous run, last line wins. Unreadable lines are ignored."""
    done: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if "jev_p_yes" in row:
                done[row_key(row)] = row
            else:
                done.pop(row_key(row), None)  # an error row supersedes nothing useful: retry it
        except (ValueError, KeyError):
            continue
    return done


def apply_threshold(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Derive ``jev_correct`` from the stored ``jev_p_yes`` at report time."""
    return [{**r, "jev_correct": r["jev_p_yes"] > threshold} if "jev_p_yes" in r else r for r in rows]


async def judge_all(items: list[dict[str, Any]], judge: Judge,
                    on_row: Callable[[dict[str, Any]], None] | None = None,
                    done: dict[tuple[str, str], dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Attach ``jev_p_yes`` to each item; failures get ``error`` and no verdict. Never raises per item.

    ``on_row`` is called with each newly finished row (used to persist progress). ``done`` holds
    verdicts from a previous run: matching rows (same key and input hash) are reused with no call.
    Items whose original judge call failed are not sent to Jev: there is nothing to compare against.
    Every task is awaited even when one fails, so nothing keeps running after this returns.
    """
    async def one(item: dict[str, Any]) -> dict[str, Any]:
        item = {**item, "input_hash": input_hash(item)}
        if item.get("original_judge_failed"):
            row = {**item, "skipped": "original judge call failed"}
        else:
            prior = (done or {}).get(row_key(item))
            if prior is not None and prior.get("input_hash") == item["input_hash"]:
                return {**item, **{k: prior[k] for k in ("jev_p_yes", "input_tokens", "output_tokens") if k in prior},
                        "resumed": True}
            try:
                v = await judge(item["category"], item["question"], item["gold"], item["generated"])
                row = {**item, "jev_p_yes": v.p_yes,
                       "input_tokens": v.input_tokens, "output_tokens": v.output_tokens}
            except Exception as exc:  # any failure of one item must not take the others down
                row = {**item, "error": f"{type(exc).__name__}: {exc}"}
        if on_row:
            on_row(row)
        return row

    results = await asyncio.gather(*(one(i) for i in items), return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):  # only on_row can get here (e.g. disk full): surface it, loudly
            raise r
    return list(results)  # type: ignore[arg-type]


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement statistics over the rows that got a verdict. Pure."""
    ok = [r for r in rows if "jev_correct" in r]
    orig = [r["original_correct"] for r in ok]
    jev = [r["jev_correct"] for r in ok]
    agree = [r for r in ok if r["original_correct"] == r["jev_correct"]]

    def rate(sub: list[dict[str, Any]]) -> float | None:
        return sum(1 for r in sub if r["original_correct"] == r["jev_correct"]) / len(sub) if sub else None

    def rate_at(sub: list[dict[str, Any]], t: float) -> float | None:
        return (sum(1 for r in sub if r["original_correct"] == (r["jev_p_yes"] > t)) / len(sub)) if sub else None

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cut: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        by_cat[f'{r["category"]} {r["category_name"]}'.strip()].append(r)
        by_cut[r["cutoff"]].append(r)
    confusion = Counter((r["original_correct"], r["jev_correct"]) for r in ok)
    return {
        "judged": len(ok), "errors": sum(1 for r in rows if "error" in r),
        "skipped_original_failed": sum(1 for r in rows if "skipped" in r),
        "skipped_inferred": sum(1 for r in rows if "skipped" in r and r.get("original_failure_inferred")),
        "resumed": sum(1 for r in rows if r.get("resumed")),
        "threshold_sweep": {str(t): rate_at(ok, t) for t in SWEEP},
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
        f'- Judged **{s["judged"]}** answers ({s["errors"]} Jev errors and '
        f'{s["skipped_original_failed"]} failed original judge calls, excluded; '
        f'{s["skipped_inferred"]} of those inferred from an empty reason because the result files '
        f'predate run.py\'s judge_failed flag)',
        f'- Agreement **{pct(s["agreement"])}**, Cohen\'s kappa **{k}** '
        "(kappa is depressed when one class dominates; read it with the confusion matrix)",
        "- Agreement by Jev threshold: "
        + ", ".join(f"p>{t}: {pct(v)}" for t, v in s["threshold_sweep"].items()),
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
    src = Path(args.predicted_dir)
    items = select_items(load_items(src), args.limit, args.seed)
    if not items:
        print(f"no judged answers found under {args.predicted_dir}", file=sys.stderr)
        return 2
    to_judge = [i for i in items if not i["original_judge_failed"]]
    if args.dry_run:
        print(f"{len(to_judge)} answers to judge, {len(items) - len(to_judge)} skipped (original judge failed).")
        if to_judge:
            first = to_judge[0]
            print("First request:")
            print(json.dumps(build_request(first["category"], first["question"], first["gold"], first["generated"]), indent=2))
        return 0
    try:
        jev_ctx = JevJudge(threshold=args.threshold, concurrency=args.concurrency, breaker=args.breaker)
    except JevError as exc:  # no API key: say so plainly instead of a traceback
        print(f"compare_judges: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else src.parent / "judge_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id(src)
    progress = out_dir / f"jev_rows_{rid}.jsonl"
    done = load_progress(progress)
    if done:
        print(f"resuming run {rid}: {len(done)} verdicts already on disk")
    with progress.open("a") as fh:
        def persist(row: dict[str, Any]) -> None:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        async with jev_ctx as jev:
            rows = await judge_all(items, jev.judge, persist, done)
    rows = apply_threshold(rows, args.threshold)
    summary = summarize(rows)
    out = out_dir / f"judge_compare_{rid}_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    out.with_suffix(".md").write_text(render_markdown(summary, args.threshold, str(args.predicted_dir)))
    print(f"wrote {out}.md and .json  (agreement {summary['agreement']}, kappa {summary['kappa']}, "
          f"{summary['errors']} errors)")
    return 0


def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predicted_dir")
    p.add_argument("--out", default=None, help="report directory (default: next to the results)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=_positive_int, default=None,
                   help="judge a deterministic sample of N answers (spread over categories, cutoffs, conversations)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed for --limit")
    p.add_argument("--concurrency", type=_positive_int, default=8)
    p.add_argument("--breaker", type=_positive_int, default=10,
                   help="stop calling Jev after this many consecutive failed judgments")
    p.add_argument("--dry-run", action="store_true")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
