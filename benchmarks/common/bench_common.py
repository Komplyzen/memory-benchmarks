"""
Shared helpers for the bench.sh / benchmark.yml scripts
======================================================

Backend names, profile and price lookup, leg directory loading, and the
harness SHA. Used by scripts/gate_estimate.py, merge_results.py,
summary_table.py, strip_for_publish.py, slice_dataset.py.

Interface (kept stable for `.github/scripts/bench.sh` in the KG repo):

- Backends: ``kg-full``, ``kg-no-spread``, ``kg-no-decay``, ``mem0-oss``,
  ``none`` (alias ``no-memory`` is normalized to ``none``).
- A leg directory is the runner's ``predicted_<project>/`` directory: one
  ``<question_id>.json`` per evaluation, runner checkpoints prefixed ``_``,
  plus a sidecar ``leg_meta.json`` written by bench.sh::

    {
      "backend": "kg-full", "shard": 0,
      "models": {"answerer_model", "answerer_provider", "judge_model",
                 "judge_provider", "agent_model", "agent_provider", "mem0_llm_model"},
      "cutoffs": [20, 100],
      "harness_sha": "<fork commit>",
      "kg_commit": "<KG commit>",             # optional
      "kg_meta": {...client.ingest_metadata()}, # optional, kg-* legs
      "wall_seconds": 1234.5,                   # optional
      "phase_failed": "ingest",                 # optional, failed legs
      "embed_failures": 0,                      # optional
      "mem0": {"llm_model": "...", "search_flags": {"rerank": false, "top_k": 100},
               "image_digest": "sha256:..."},   # optional, mem0-oss legs
      "versions": {"docker": "...", "bun": "...", "python": "..."}  # optional
    }
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.common.utils import FULL_CONTEXT_CUTOFF, cutoff_label

KG_BACKENDS = ("kg-full", "kg-no-spread", "kg-no-decay")
BACKENDS = KG_BACKENDS + ("mem0-oss", "none")
BACKEND_ALIASES = {"no-memory": "none", "kg": "kg-full", "oss": "mem0-oss"}
KG_QUERY_LIMIT_CAP = 100

BACKEND_FLAGS = {
    "kg-full": ["--backend", "kg", "--kg-spread", "on", "--kg-decay", "on"],
    "kg-no-spread": ["--backend", "kg", "--kg-spread", "off", "--kg-decay", "on"],
    "kg-no-decay": ["--backend", "kg", "--kg-spread", "on", "--kg-decay", "off"],
    "mem0-oss": ["--backend", "oss"],
    "none": ["--backend", "none"],
}

BACKEND_VARIANT = {"kg-full": "full", "kg-no-spread": "no-spread", "kg-no-decay": "no-decay"}

PROFILES_PATH = Path(__file__).with_name("profiles.json")


class BenchError(Exception):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# Backends / cutoffs
# ---------------------------------------------------------------------------


def normalize_backends(csv: str) -> list[str]:
    out: list[str] = []
    for raw in csv.split(","):
        name = raw.strip()
        if not name:
            continue
        name = BACKEND_ALIASES.get(name, name)
        if name not in BACKENDS:
            raise BenchError(f"unknown backend {raw.strip()!r}; valid: {', '.join(BACKENDS)}", 2)
        if name not in out:
            out.append(name)
    if not out:
        raise BenchError("no backends given", 2)
    return out


def normalize_cutoffs(csv: str | list[int], backends: list[str] | None = None) -> list[int]:
    if isinstance(csv, str):
        try:
            values = [int(c.strip()) for c in csv.split(",") if c.strip()]
        except ValueError as exc:
            raise BenchError(f"cutoffs must be integers: {csv!r}", 2) from exc
    else:
        values = [int(c) for c in csv]
    if not values or any(v <= 0 for v in values):
        raise BenchError(f"cutoffs must be positive integers: {csv!r}", 2)
    cutoffs = sorted(set(values))
    if backends and any(b in KG_BACKENDS for b in backends):
        too_big = [c for c in cutoffs if c > KG_QUERY_LIMIT_CAP]
        if too_big:
            raise BenchError(f"cutoffs {too_big} exceed the kg_query limit cap of {KG_QUERY_LIMIT_CAP} for kg-* backends", 2)
    return cutoffs


def cutoffs_for_backend(backend: str, cutoffs: list[int]) -> list[int]:
    """--backend none is evaluated at the single pseudo-cutoff full_context."""
    return [FULL_CONTEXT_CUTOFF] if backend == "none" else list(cutoffs)


# ---------------------------------------------------------------------------
# Profiles / prices
# ---------------------------------------------------------------------------


def load_profiles(path: Path | None = None) -> dict[str, Any]:
    with open(path or PROFILES_PATH) as f:
        return json.load(f)


MODEL_KEYS = ("answerer_model", "answerer_provider", "judge_model", "judge_provider",
              "agent_model", "agent_provider", "mem0_llm_model")


def resolve_models(profile_name: str, overrides: dict[str, str | None], profiles: dict[str, Any] | None = None) -> dict[str, str]:
    profiles = profiles or load_profiles()
    if profile_name not in profiles["profiles"]:
        raise BenchError(f"unknown profile {profile_name!r}; valid: {', '.join(profiles['profiles'])}", 2)
    prof = profiles["profiles"][profile_name]
    models = {k: prof[k] for k in MODEL_KEYS}
    for k, v in overrides.items():
        if v:
            models[k] = v
    return models


def price_for(model: str, profiles: dict[str, Any]) -> dict[str, float | None] | None:
    return profiles.get("prices", {}).get(model)


def usd_cost(model: str, profiles: dict[str, Any], in_tokens: float, out_tokens: float, cached_tokens: float = 0.0) -> float | None:
    p = price_for(model, profiles)
    if p is None:
        return None
    cached_price = p.get("cached_input_per_m_usd")
    if cached_price is None:  # no cache discount: cached tokens billed as input
        in_tokens += cached_tokens
        cached_tokens = 0.0
    return (in_tokens * p["input_per_m_usd"] + cached_tokens * (cached_price or 0.0) + out_tokens * p["output_per_m_usd"]) / 1_000_000


# ---------------------------------------------------------------------------
# Leg directories
# ---------------------------------------------------------------------------

LEG_META_NAMES = ("leg_meta.json", "_leg_meta.json")


def load_leg(dir_path: str | Path) -> dict[str, Any]:
    """Load one predicted_<project>/ directory: evaluations + leg_meta."""
    d = Path(dir_path)
    if not d.is_dir():
        raise BenchError(f"{d}: not a directory", 2)
    meta = None
    for name in LEG_META_NAMES:
        p = d / name
        if p.exists():
            with open(p) as f:
                meta = json.load(f)
            break
    if meta is None:
        raise BenchError(f"{d}: no leg_meta.json sidecar (bench.sh writes it next to the runner's per-question files)", 2)
    evaluations: list[dict[str, Any]] = []
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_") or p.name in LEG_META_NAMES:
            continue
        with open(p) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise BenchError(f"{p}: invalid JSON ({exc})", 2) from exc
        if isinstance(data, dict) and "question_id" in data:
            evaluations.append(data)
    return {"dir": str(d), "meta": meta, "evaluations": evaluations}


def harness_sha(repo_dir: str | Path | None = None) -> str | None:
    repo = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=5,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.getenv("HARNESS_SHA")


# ---------------------------------------------------------------------------
# Evaluation-derived numbers (shared by merge and summary)
# ---------------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    return len(text) // 4


def tokens_to_answerer(evaluations: list[dict[str, Any]], cutoff: int) -> float | None:
    """Mean approx tokens of memory text handed to the answerer at a cutoff."""
    vals = []
    for e in evaluations:
        results = (e.get("retrieval") or {}).get("search_results") or []
        vals.append(sum(approx_tokens(r.get("memory", "") or "") for r in results[:cutoff]))
    return round(statistics.mean(vals), 1) if vals else None


def latency_percentiles(evaluations: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    lat = sorted(float((e.get("retrieval") or {}).get("search_latency_ms") or 0.0) for e in evaluations)
    if not lat:
        return None, None
    def pct(p: float) -> float:
        k = max(0, min(len(lat) - 1, int(round(p * (len(lat) - 1)))))
        return round(lat[k], 1)
    return pct(0.5), pct(0.95)


def answer_judge_calls(evaluations: list[dict[str, Any]]) -> int:
    """Each evaluated cutoff costs one answerer and one judge call."""
    return sum(2 * len(e.get("cutoff_results") or {}) for e in evaluations)


def labels_for(cutoffs: list[int]) -> list[str]:
    return [cutoff_label(c) for c in cutoffs]
