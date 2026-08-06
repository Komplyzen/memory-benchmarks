"""The generic scoring seam: run an arbitrary scorer over a run's artifacts.

The machine knows nothing about terms, sessions, or relevance. A scorer is a
script (typically derived by the eval-operator, approved by the human, saved
under scorers/<benchmark>/). Conductor's whole job here:

  1. hand the scorer a context JSON (run row, result file, predict dir,
     dataset path, config, origin);
  2. take back a JSON verdict {mode, metrics, validity, headline?};
  3. record it in the ledger under metrics_json["derived_modes"][mode],
     permanently tagged with its mode + validity note so a directional number
     can never masquerade as a canonical one.

Contract for scorers:
  invoked as: python <scorer.py> <context.json>
  stdout (last line must be JSON): {"mode": str, "metrics": {name: number},
    "validity": str, "headline": number|null, "headline_label": str|null}
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import db
from .paths import repo_root, run_dir


def score_run(run_id: str, scorer_path: str, timeout: float = 600.0) -> dict[str, Any]:
    row = db.get_run(run_id)
    if not row:
        raise ValueError(f"no such run {run_id!r}")
    origin_row = db.get_origin(run_id)
    origin = json.loads(origin_row["origin_json"]) if origin_row else {}
    config = json.loads(row["config"] or "{}")

    scorer = Path(scorer_path).expanduser().resolve()
    if not scorer.is_file():
        raise FileNotFoundError(f"scorer not found: {scorer}")

    # Convention across benchmark runners: results/<benchmark>/predicted_<project>.
    predict_dir = (
        repo_root() / "results" / row["template_id"] / f"predicted_{row['project_name']}"
    )
    context = {
        "run_id": run_id,
        "benchmark": row["template_id"],
        "project_name": row["project_name"],
        "config": config,
        "origin": origin,
        "result_file": row["result_file"],
        "predict_dir": str(predict_dir) if predict_dir.is_dir() else None,
        "dataset_path": config.get("dataset_path")
        or (origin.get("dataset") or {}).get("path"),
        "repo_root": str(repo_root()),
    }
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    ctx_path = rd / f"score_context_{scorer.stem}.json"
    ctx_path.write_text(json.dumps(context, indent=2))

    proc = subprocess.run(
        [sys.executable, str(scorer), str(ctx_path)],
        capture_output=True, text=True, timeout=timeout, cwd=str(repo_root()),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"scorer failed (exit {proc.returncode}): {proc.stderr.strip()[:400]}"
        )
    last = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    if not last:
        raise RuntimeError("scorer produced no output")
    verdict = json.loads(last[-1])
    for field in ("mode", "metrics", "validity"):
        if field not in verdict:
            raise RuntimeError(f"scorer verdict missing required field {field!r}")

    # Record: tagged, alongside (never replacing) canonical metrics.
    existing = {}
    if origin_row is not None:
        keys = origin_row.keys()
        if "metrics_json" in keys and origin_row["metrics_json"]:
            try:
                existing = json.loads(origin_row["metrics_json"])
            except Exception:
                existing = {}
    modes = existing.get("derived_modes") or {}
    modes[verdict["mode"]] = {
        "metrics": verdict["metrics"],
        "validity": verdict["validity"],
        "headline": verdict.get("headline"),
        "headline_label": verdict.get("headline_label"),
        "scorer": str(scorer),
        "scored_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    existing["derived_modes"] = modes
    db.set_metrics(run_id, existing)
    return verdict
