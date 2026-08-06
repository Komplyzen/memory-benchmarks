"""conductor CLI: start / status / logs / ls / stop.

No approval gate -- by the time a command runs, trust was established upstream
(the agent<->user layer). The CLI executes unconditionally and records what happened.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import sys
from typing import Any

from . import db, launcher, materialize as materialize_mod, shelf
from .preflight import PreflightError


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _coerce(s: str) -> Any:
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_kv(pairs: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            sys.exit(f"conductor: {what} must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v
    return out


def _benchmark_of(row) -> str:
    return row["template_id"] if row else "?"


# --- commands ---


def cmd_start(args: argparse.Namespace) -> int:
    raw = _parse_kv(args.set or [], "--set")
    config = {k.replace("-", "_"): _coerce(v) for k, v in raw.items()}
    env_overrides = _parse_kv(args.env or [], "--env")
    describe = _parse_kv(args.describe or [], "--describe")
    if args.name:
        describe["name"] = args.name

    try:
        run_id = launcher.start(
            benchmark=args.benchmark,
            config=config,
            env_overrides=env_overrides,
            project_name=args.project_name,
            note=args.note,
            describe=describe or None,
            skip_preflight=args.skip_preflight,
            reuse=args.reuse,
        )
    except PreflightError as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 2
    print(run_id)
    return 0


def _origin_metrics(origin) -> dict:
    try:
        keys = origin.keys() if origin else []
        if origin and "metrics_json" in keys and origin["metrics_json"]:
            return json.loads(origin["metrics_json"])
    except Exception:
        pass
    return {}


def cmd_ls(args: argparse.Namespace) -> int:
    rows = db.list_runs(status=args.status, limit=args.n)
    if not rows:
        print("(no runs)")
        return 0
    print(f"{'RUN':<14} {'BENCHMARK':<12} {'STATUS':<10} {'RESULT':<14} {'CREATED':<20} NOTE")
    for r in rows:
        origin = db.get_origin(r["id"])
        note = (origin["note"] if origin and origin["note"] else "") or ""
        m = _origin_metrics(origin)
        pairs = list((m.get("pairs") or {}).items())
        acc = f"{pairs[0][0]} {pairs[0][1]}" if pairs else ""
        print(
            f"{r['id']:<14} {_benchmark_of(r):<12} {r['status']:<10} {acc:<14} "
            f"{(r['created_at'] or ''):<20} {note[:36]}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    r = db.get_run(args.run_id)
    if not r:
        sys.exit(f"conductor: no such run {args.run_id!r}")
    origin_row = db.get_origin(args.run_id)
    origin = json.loads(origin_row["origin_json"]) if origin_row else {}

    print(f"run        {r['id']}")
    print(f"benchmark  {_benchmark_of(r)}")
    print(f"status     {r['status']}")
    print(f"project    {r['project_name']}")
    print(f"target     {origin.get('target')}")
    print(f"created    {r['created_at']}")
    print(f"started    {r['started_at']}")
    print(f"finished   {r['finished_at']}")
    print(f"pid        {r['pid']}")
    if origin.get("note"):
        print(f"note       {origin['note']}")
    git = (origin.get("git") or {}).get("benchmark_repo") or {}
    if git:
        dirty = " (dirty)" if git.get("dirty") else ""
        print(f"git        {git.get('branch')} @ {(git.get('commit') or '')[:10]}{dirty}")
    ds = origin.get("dataset")
    if ds and ds.get("sha256"):
        print(f"dataset    {ds['path']}  sha={ds['sha256'][:12]}")
    print(f"config     {r['config']}")
    m = _origin_metrics(origin_row)
    if m.get("by_cutoff"):
        cuts = "  ".join(f"{k}={v:.1f}" for k, v in sorted(m["by_cutoff"].items()))
        print(f"accuracy   {cuts}   (n={m.get('total_questions')})")
    if m.get("INTEGRITY_WARNING"):
        print(f"!! WARNING  {m['INTEGRITY_WARNING']}")
    for mode, dm in (m.get("derived_modes") or {}).items():
        vals = "  ".join(f"{k}={v}" for k, v in list(dm.get("metrics", {}).items())[:6])
        print(f"derived    [{mode}] {vals}")
        print(f"           validity: {dm.get('validity')}")
    if origin.get("describe", {}).get("reuse_store"):
        print(f"reuse      {origin['describe']['reuse_store']}")
    print(f"log        {r['log_file']}")
    if r["result_file"]:
        print(f"result     {r['result_file']}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    r = db.get_run(args.run_id)
    if not r:
        sys.exit(f"conductor: no such run {args.run_id!r}")
    if not r["log_file"] or not os.path.isfile(r["log_file"]):
        print("(no log yet)")
        return 0
    with open(r["log_file"], errors="replace") as f:
        lines = f.readlines()
    for line in lines[-args.n :]:
        sys.stdout.write(line)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    r = db.get_run(args.run_id)
    if not r:
        sys.exit(f"conductor: no such run {args.run_id!r}")
    if r["status"] not in ("running", "pending"):
        print(f"{args.run_id} is {r['status']}; nothing to stop")
        return 0
    # Mark stopped first so the supervisor's exit handler won't relabel it failed.
    db.update_run(args.run_id, status="stopped", finished_at=_utc_now_iso())
    pid = r["pid"]
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # already gone
        except Exception as e:
            print(f"warning: could not signal pid {pid}: {e}", file=sys.stderr)
    print(f"stopped {args.run_id}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    rows = []
    for rid in (args.run_a, args.run_b):
        r = db.get_run(rid)
        if not r:
            sys.exit(f"conductor: no such run {rid!r}")
        o = db.get_origin(rid)
        origin = json.loads(o["origin_json"]) if o else {}
        rows.append({"run": r, "origin": origin, "metrics": _origin_metrics(o)})
    a, b = rows

    print(f"diff {args.run_a} -> {args.run_b}\n")
    # config delta (flat dicts from --set)
    ca = json.loads(a["run"]["config"] or "{}")
    cb = json.loads(b["run"]["config"] or "{}")
    keys = sorted(set(ca) | set(cb))
    changes = [(k, ca.get(k), cb.get(k)) for k in keys if ca.get(k) != cb.get(k)]
    if changes:
        print("config:")
        for k, va, vb in changes:
            print(f"  {k}: {va} -> {vb}")
    else:
        print("config: identical")
    # shared foundations
    ra = a["origin"].get("describe", {}).get("reuse_store")
    rb = b["origin"].get("describe", {}).get("reuse_store")
    if ra or rb:
        print(f"store:  {ra} {'== shared' if ra == rb and ra else '-> ' + str(rb)}")
    ta, tb = a["origin"].get("target"), b["origin"].get("target")
    if ta != tb:
        print(f"target: {ta} -> {tb}")
    # metric deltas
    ma, mb = a["metrics"].get("by_cutoff", {}), b["metrics"].get("by_cutoff", {})
    if ma or mb:
        print("accuracy:")
        for label in sorted(set(ma) | set(mb)):
            va, vb = ma.get(label), mb.get(label)
            if va is not None and vb is not None:
                print(f"  {label}: {va:.1f} -> {vb:.1f}  ({vb - va:+.1f})")
            else:
                print(f"  {label}: {va} -> {vb}")
    else:
        print("accuracy: (no metrics yet on one or both runs)")
    # Derived (mode-tagged) metrics: compare only like-with-like, always labeled.
    dma = a["metrics"].get("derived_modes") or {}
    dmb = b["metrics"].get("derived_modes") or {}
    for mode in sorted(set(dma) & set(dmb)):
        ma2, mb2 = dma[mode].get("metrics", {}), dmb[mode].get("metrics", {})
        print(f"derived [{mode}] (directional — {dma[mode].get('validity','')[:80]}):")
        for k in sorted(set(ma2) & set(mb2)):
            va, vb = ma2[k], mb2[k]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                print(f"  {k}: {va} -> {vb}  ({vb - va:+.2f})")
    for mode in sorted(set(dma) ^ set(dmb)):
        print(f"derived [{mode}]: only one run has it — not comparable")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from . import score as score_mod
    v = score_mod.score_run(args.run_id, args.scorer)
    print(f"mode      {v['mode']}  (DERIVED — {v['validity']})")
    for k, val in v["metrics"].items():
        print(f"  {k}: {val}")
    if v.get("headline") is not None:
        print(f"headline  {v['headline']} ({v.get('headline_label')})")
    print("recorded in ledger under derived_modes (tagged; canonical metrics untouched)")
    return 0


def cmd_verify_store(args: argparse.Namespace) -> int:
    """Evidence, not verdict: instruments for the operator's open-ended review."""
    from . import verify
    res = verify.verify_store(args.artifact_id, sample=args.sample, qdrant=args.qdrant)
    for c in res["checks"]:
        mark = " ok " if c["ok"] else "FLAG"
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    if res["passed"]:
        print("EVIDENCE ONLY: no red flags on these instruments. This is NOT a "
              "verdict -- an operator review (intent + expected behavior) still decides.")
    else:
        print("RED FLAGS FOUND: strong evidence the setup is wrong. Investigate "
              "before any eval stands on this store.")
    return 0 if res["passed"] else 1


def cmd_shelf_ls(args: argparse.Namespace) -> int:
    rows = db.list_artifacts(benchmark=args.benchmark, layer=args.layer)
    if not rows:
        print("(no artifacts)")
        return 0
    print(f"{'ARTIFACT':<16} {'BENCHMARK':<12} {'LAYER':<20} {'QUESTIONS':<10} CREATED")
    for r in rows:
        origin = json.loads(r["origin_json"])
        qc = origin.get("question_count", "?")
        print(f"{r['id']:<16} {r['benchmark']:<12} {r['layer']:<20} {str(qc):<10} {r['created_at']}")
    return 0


def cmd_shelf_show(args: argparse.Namespace) -> int:
    m = shelf.load_manifest(args.artifact_id)
    if not m:
        sys.exit(f"conductor: no such artifact {args.artifact_id!r}")
    print(f"artifact    {m['artifact_id']}")
    print(f"benchmark   {m['benchmark']}")
    print(f"layer       {m['layer']}")
    print(f"adopted     {m.get('adopted')}")
    print(f"questions   {m.get('question_count')}")
    print(f"memories    {m.get('memory_count')}")
    print(f"extraction  {m.get('extraction_model')}")
    ds = m.get("dataset") or {}
    if ds.get("sha256"):
        print(f"dataset     {ds.get('path')}  sha={ds['sha256'][:12]}")
    srj = m.get("source_run_json") or {}
    if srj:
        print(f"platform    branch={srj.get('platform_branch')} head={(srj.get('platform_head') or '')[:10]}")
    if m.get("note"):
        print(f"note        {m['note']}")
    print("caveats:")
    for c in m.get("caveats", []):
        print(f"  - {c}")
    print(f"stats       {json.dumps(m.get('stats', {}))}")
    return 0


def cmd_shelf_adopt(args: argparse.Namespace) -> int:
    m = shelf.adopt_longmemeval_dump(
        source_dir=args.source_dir,
        dataset_path=args.dataset,
        artifact_id=args.id,
        note=args.note,
        max_questions=args.max_questions,
        progress=lambda s: print(f"  {s}", file=sys.stderr),
    )
    print(m["artifact_id"])
    print(
        f"adopted {m['question_count']} questions / {m['memory_count']} memories "
        f"(model={m.get('extraction_model')})",
        file=sys.stderr,
    )
    return 0


def cmd_shelf_adopt_browsecomp(args: argparse.Namespace) -> int:
    from . import browsecomp

    manifest = browsecomp.adopt_store(
        manifest_path=args.manifest,
        host=args.host,
        corpus_path=args.corpus,
        queries_path=args.queries,
        qrels_path=args.qrels,
        gold_qrels_path=args.gold_qrels,
        artifact_id=args.id,
        note=args.note,
    )
    print(manifest["artifact_id"])
    return 0


def cmd_materialize(args: argparse.Namespace) -> int:
    try:
        m = materialize_mod.materialize(
            artifact_id=args.artifact_id,
            host=args.host,
            store_id=args.store_id,
            max_questions=args.max_questions,
            question_ids=(
                [q.strip() for q in args.question_ids.split(",") if q.strip()]
                if args.question_ids else None
            ),
            max_workers=args.max_workers,
            note=args.note,
            progress=lambda s: print(f"  {s}", file=sys.stderr),
        )
    except PreflightError as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 2
    except materialize_mod.MaterializeAborted as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 2
    print(m["artifact_id"])
    print(
        f"materialized {m['question_count']} questions / {m['stored_memory_count']} memories "
        f"(dedup dropped {m.get('dedup_dropped')}) on {m.get('embedder')} @ {m['host']}",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="conductor", description="Run and record memory-system evals.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("start", help="Launch a benchmark run (detached).")
    s.add_argument("benchmark", choices=sorted(launcher.BENCHMARKS))
    s.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="Pipeline config override (repeatable), e.g. --set top_k=30")
    s.add_argument("--env", action="append", metavar="KEY=VALUE",
                   help="Environment override for the run (repeatable)")
    s.add_argument("--name", default=None,
                   help="Short human name for the experiment (shown in UIs; e.g. bm25-detune-1.4).")
    s.add_argument("--project-name", default=None, help="Defaults to the run id.")
    s.add_argument("--note", default=None, help="Freeform origin-story note.")
    s.add_argument("--describe", action="append", metavar="KEY=VALUE",
                   help="Structured origin-story facts (repeatable).")
    s.add_argument("--skip-preflight", action="store_true",
                   help="Skip the embedder/platform preflight (not recommended).")
    s.add_argument("--reuse", default=None, metavar="STORE_ARTIFACT",
                   help="Reuse a materialized memory_store: skip ingestion, run search/answer/judge over it.")
    s.set_defaults(func=cmd_start)

    ls_parser = sub.add_parser("ls", help="List recent runs.")
    ls_parser.add_argument("--status", default=None,
                           choices=["pending", "running", "succeeded", "failed", "stopped"])
    ls_parser.add_argument("-n", type=int, default=20)
    ls_parser.set_defaults(func=cmd_ls)

    st = sub.add_parser("status", help="Show one run's record + origin story.")
    st.add_argument("run_id")
    st.set_defaults(func=cmd_status)

    lg = sub.add_parser("logs", help="Tail a run's log.")
    lg.add_argument("run_id")
    lg.add_argument("-n", type=int, default=200)
    lg.set_defaults(func=cmd_logs)

    sp = sub.add_parser("stop", help="Stop a running run.")
    sp.add_argument("run_id")
    sp.set_defaults(func=cmd_stop)

    df = sub.add_parser("diff", help="Compare two runs: config delta, shared store, metric deltas.")
    df.add_argument("run_a")
    df.add_argument("run_b")
    df.set_defaults(func=cmd_diff)

    sh = sub.add_parser("shelf", help="Inspect/populate the artifact shelf.")
    shsub = sh.add_subparsers(dest="shelf_command", required=True)

    shls = shsub.add_parser("ls", help="List shelved artifacts.")
    shls.add_argument("--benchmark", default=None)
    shls.add_argument("--layer", default=None)
    shls.set_defaults(func=cmd_shelf_ls)

    shshow = shsub.add_parser("show", help="Show one artifact's manifest.")
    shshow.add_argument("artifact_id")
    shshow.set_defaults(func=cmd_shelf_show)

    shad = shsub.add_parser("adopt", help="Import an existing extraction dump onto the shelf.")
    shad.add_argument("source_dir", help="Path to the per-turn extraction dump (has cycle_*/).")
    shad.add_argument("--dataset", required=True, help="Dataset JSON whose question_ids to map to.")
    shad.add_argument("--id", default=None, help="Artifact id (default: emx-<hex>).")
    shad.add_argument("--note", default=None)
    shad.add_argument("--max-questions", type=int, default=None,
                      help="Cap distinct questions (smoke adopts).")
    shad.set_defaults(func=cmd_shelf_adopt)

    shbcp = shsub.add_parser("adopt-browsecomp", help="Adopt a completed platform BrowseComp store manifest.")
    shbcp.add_argument("--manifest", required=True)
    shbcp.add_argument("--host", required=True)
    shbcp.add_argument("--corpus", required=True)
    shbcp.add_argument("--queries", required=True)
    shbcp.add_argument("--qrels", required=True)
    shbcp.add_argument("--gold-qrels", required=True)
    shbcp.add_argument("--id", default=None)
    shbcp.add_argument("--note", default=None)
    shbcp.set_defaults(func=cmd_shelf_adopt_browsecomp)

    mz = sub.add_parser("materialize",
                        help="Replay an extracted-memories artifact into a live faithful store.")
    mz.add_argument("artifact_id")
    mz.add_argument("--host", required=True, help="Target mem0 server (e.g. http://localhost:8888).")
    mz.add_argument("--store-id", default=None, help="Store id / run tag (default: ms<hex>).")
    mz.add_argument("--max-questions", type=int, default=None, help="Cap questions (smoke).")
    mz.add_argument("--question-ids", default=None, help="Comma-separated question_ids to materialize.")
    mz.add_argument("--max-workers", type=int, default=8)
    mz.add_argument("--note", default=None)
    mz.set_defaults(func=cmd_materialize)

    sc = sub.add_parser("score",
                        help="Run a scorer over a run's artifacts; record mode-tagged metrics.")
    sc.add_argument("run_id")
    sc.add_argument("--scorer", required=True, help="Path to scorer script (see conductor/score.py contract).")
    sc.set_defaults(func=cmd_score)

    vs = sub.add_parser("verify-store",
                        help="Mechanical faithfulness audit of a materialized store (exit 1 on FAIL).")
    vs.add_argument("artifact_id")
    vs.add_argument("--sample", type=int, default=8)
    vs.add_argument("--qdrant", default="http://localhost:6333")
    vs.set_defaults(func=cmd_verify_store)

    return p


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
