"""Start a run: build the command, record it, and hand it to a detached supervisor.

`start()` returns a run id immediately and never blocks. The actual pipeline runs
under a supervisor process in its own session (contract #2 -- runs are detached and
addressed by id, so they outlive the terminal or agent session that launched them).
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from . import db, preflight, record, reuse as reuse_mod
from .paths import repo_root, run_dir

# benchmark -> (template_id, module, run-id prefix). template_id mirrors
# db._SEED_TEMPLATES; module is the `-m` target (run as a module so the
# `benchmarks.*` package imports resolve against the repo root, not the script dir).
BENCHMARKS: dict[str, tuple[str, str, str]] = {
    "longmemeval": ("longmemeval", "benchmarks.longmemeval.run", "lme"),
    "locomo": ("locomo", "benchmarks.locomo.run", "locomo"),
    "beam": ("beam", "benchmarks.beam.run", "beam"),
    "browsecomp_plus": ("browsecomp_plus", "benchmarks.browsecomp_plus.run", "bcp"),
}


def build_args(config: dict[str, Any]) -> list[str]:
    """dict -> CLI flags, matching src/lib/executor.ts buildArgs() exactly.

    {max_workers: 10, debug: True} -> ["--max-workers", "10", "--debug"]
    """
    args: list[str] = []
    for key, value in config.items():
        if value is None or value == "" or value is False:
            continue
        if isinstance(value, list) and not value:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif isinstance(value, list):
            args.extend([flag, ",".join(str(v) for v in value)])
        else:
            args.extend([flag, str(value)])
    return args


def _new_run_id(prefix: str) -> str:
    for _ in range(10):
        rid = f"{prefix}-{secrets.token_hex(2)}"
        if not run_dir(rid).exists():
            return rid
    raise RuntimeError("could not allocate a unique run id")


def start(
    *,
    benchmark: str,
    config: dict[str, Any],
    env_overrides: Optional[dict[str, str]] = None,
    project_name: Optional[str] = None,
    note: Optional[str] = None,
    describe: Optional[dict[str, str]] = None,
    skip_preflight: bool = False,
    reuse: Optional[str] = None,
) -> str:
    if benchmark not in BENCHMARKS:
        raise ValueError(
            f"unknown benchmark {benchmark!r}; known: {', '.join(sorted(BENCHMARKS))}"
        )
    template_id, module, prefix = BENCHMARKS[benchmark]
    env_overrides = env_overrides or {}
    describe = dict(describe or {})

    # Reuse: point the run at a materialized store's host + dataset. Checkpoints
    # (written below, once the run id is known) make run.py skip ingestion.
    reuse_meta: Optional[dict[str, Any]] = None
    if reuse:
        store = reuse_mod.load_store(reuse)
        sm = store["manifest"]
        if sm.get("benchmark") != benchmark:
            raise ValueError(f"store {reuse} belongs to {sm.get('benchmark')}, not {benchmark}")
        if benchmark == "browsecomp_plus":
            files = sm.get("benchmark_files") or {}
            config["mem0_host"] = sm["host"]
            config["store_manifest"] = str(Path(store["path"]) / "manifest.json")
            config["corpus"] = sm["corpus"]["path"]
            config["corpus_revision"] = sm["corpus"]["revision"]
            config["queries"] = files["queries"]
            config["qrels"] = files["qrels"]
            config["gold_qrels"] = files["gold_qrels"]
        else:
            config.setdefault("backend", "oss")
            config["mem0_host"] = sm.get("host")
            if sm.get("dataset", {}).get("path"):
                config.setdefault("dataset_path", sm["dataset"]["path"])

    # Fail loud BEFORE recording anything if the embedder/platform is unusable.
    if not skip_preflight:
        preflight.require_ok(config, env_overrides)

    run_id = _new_run_id(prefix)
    project_name = project_name or run_id

    if reuse:
        if benchmark == "browsecomp_plus":
            reuse_meta = {
                "store_artifact": reuse,
                "host": sm["host"],
                "documents": sm["corpus"]["document_count"],
                "corpus_revision": sm["corpus"]["revision"],
            }
        else:
            reuse_meta = reuse_mod.write_checkpoints(reuse, project_name)
        describe["reuse_store"] = reuse
        describe["reuse_rationale"] = (
            f"Reusing BrowseComp store {reuse} with {reuse_meta['documents']} fixed documents; only search mode varies."
            if benchmark == "browsecomp_plus"
            else (
                f"Reusing materialized store {reuse} (run_id={reuse_meta['store_run_id']}, "
                f"embedder={reuse_meta['embedder']}); run.py skips ingestion for "
                f"{reuse_meta['questions']} questions via pre-written checkpoints. "
                f"Zero extraction spend; only search/answer/judge run."
            )
        )

    argv = [sys.executable, "-m", module, "--project-name", project_name, *build_args(config)]

    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    log_file = str(rd / "run.log")

    origin = record.capture_origin(
        benchmark=benchmark,
        argv=argv,
        config=config,
        env_overrides=env_overrides,
        note=note,
        describe=describe,
        dataset_path=config.get("dataset_path"),
    )
    (rd / "origin.json").write_text(json.dumps(origin, indent=2))

    launch_spec = {
        "run_id": run_id,
        "argv": argv,
        "cwd": str(repo_root()),
        "env_overrides": env_overrides,
        "log_file": log_file,
    }
    (rd / "launch.json").write_text(json.dumps(launch_spec, indent=2))

    # Ledger rows first, so the run is visible even if the supervisor spawn races.
    db.insert_run(run_id, template_id, project_name, config, env_overrides)
    db.update_run(run_id, log_file=log_file)
    db.insert_origin(run_id, benchmark, origin.get("target"), note, origin)

    # Detached supervisor: own session (setsid) + no controlling terminal, so it
    # survives us. It flips the run to running/succeeded/failed on its own.
    subprocess.Popen(
        [sys.executable, "-m", "conductor.supervisor", run_id],
        cwd=str(repo_root()),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return run_id
