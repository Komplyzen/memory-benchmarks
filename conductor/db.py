"""SQLite ledger, shared with the Next.js web UI.

The two web-owned tables (`eval_templates`, `eval_runs`) are created here with the
IDENTICAL schema the UI uses (`src/lib/db.ts`) so either side can create the DB
first and the other is happy. We add `run_origin` for the freeform origin story
that the conceptual model requires but the UI never had a place for.

WAL + busy_timeout let the Python CLI write while a `next dev` server holds the
same file open.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from .paths import db_path

# Kept byte-for-byte in step with src/lib/db.ts + src/lib/templates.ts seeds so the
# FK (eval_runs.template_id -> eval_templates.id) resolves regardless of who creates
# the DB. INSERT OR IGNORE means we never clobber the UI's seeds.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_templates (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  eval_type TEXT NOT NULL,
  script_path TEXT NOT NULL,
  description TEXT,
  default_config TEXT DEFAULT '{}',
  default_eval_config TEXT DEFAULT '{}',
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY,
  template_id TEXT NOT NULL REFERENCES eval_templates(id),
  project_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  config TEXT DEFAULT '{}',
  env_overrides TEXT DEFAULT '{}',
  pid INTEGER,
  log_file TEXT,
  result_file TEXT,
  started_at TEXT,
  finished_at TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON eval_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_template ON eval_runs(template_id);

-- Conductor-only: the origin story (how the world was set up when this run ran).
-- launched_by distinguishes CLI/agent runs from web-UI runs in one ledger.
CREATE TABLE IF NOT EXISTS run_origin (
  run_id TEXT PRIMARY KEY REFERENCES eval_runs(id) ON DELETE CASCADE,
  benchmark TEXT,
  target TEXT,
  note TEXT,
  launched_by TEXT DEFAULT 'conductor',
  captured_at TEXT DEFAULT (datetime('now')),
  origin_json TEXT NOT NULL
);

-- The shelf: reusable artifacts (a rung of the ladder). v1 layer is
-- 'extracted_memories'; later: search / answers / judgments. `created_by_run`
-- is NULL for adopted artifacts (imported, not produced here). `path` points at
-- the on-disk payload under data_root; `origin_json` carries the origin story,
-- caveats, and counts an agent reads to judge reuse.
CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  benchmark TEXT NOT NULL,
  layer TEXT NOT NULL,
  dataset_sha TEXT,
  path TEXT NOT NULL,
  created_by_run TEXT REFERENCES eval_runs(id) ON DELETE SET NULL,
  origin_json TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_artifacts_layer ON artifacts(benchmark, layer);
"""

# Mirror of SEED_TEMPLATES in src/lib/templates.ts (id + script_path are the parts
# the FK and launcher depend on; the rest is display metadata).
_SEED_TEMPLATES = [
    ("locomo", "LOCOMO-10", "benchmark", "benchmarks/locomo/run.py",
     "LOCOMO-10 benchmark."),
    ("longmemeval", "LongMemEval", "benchmark", "benchmarks/longmemeval/run.py",
     "LongMemEval-S benchmark -- 500 questions, 6 types, full haystack."),
    ("beam", "BEAM", "benchmark", "benchmarks/beam/run.py",
     "BEAM benchmark (ICLR 2026)."),
    ("browsecomp_plus", "BrowseComp-Plus", "benchmark", "benchmarks/browsecomp_plus/run.py",
     "Document-level retrieval across regular, fast, and agentic search."),
]

_initialized = False


def connect() -> sqlite3.Connection:
    """Open the ledger, ensuring schema + template seeds exist exactly once."""
    global _initialized
    conn = sqlite3.connect(str(db_path()), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    if not _initialized:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT OR IGNORE INTO eval_templates "
            "(id, name, eval_type, script_path, description) VALUES (?, ?, ?, ?, ?)",
            _SEED_TEMPLATES,
        )
        # Migration: metrics lifted from result files (added after v0.1 tables shipped).
        try:
            conn.execute("ALTER TABLE run_origin ADD COLUMN metrics_json TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
        _initialized = True
    return conn


# --- run row helpers ---


def insert_run(
    run_id: str,
    template_id: str,
    project_name: str,
    config: dict[str, Any],
    env_overrides: dict[str, str],
) -> None:
    conn = connect()
    with conn:
        conn.execute(
            "INSERT INTO eval_runs (id, template_id, project_name, status, config, env_overrides) "
            "VALUES (?, ?, ?, 'pending', ?, ?)",
            (run_id, template_id, project_name, json.dumps(config), json.dumps(env_overrides)),
        )
    conn.close()


def update_run(run_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn = connect()
    with conn:
        conn.execute(f"UPDATE eval_runs SET {cols} WHERE id = ?", (*fields.values(), run_id))
    conn.close()


def get_run(run_id: str) -> Optional[sqlite3.Row]:
    conn = connect()
    row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    return row


def list_runs(status: Optional[str] = None, limit: int = 20) -> list[sqlite3.Row]:
    conn = connect()
    if status:
        rows = conn.execute(
            "SELECT * FROM eval_runs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return rows


def insert_origin(
    run_id: str, benchmark: str, target: Optional[str], note: Optional[str], origin: dict[str, Any]
) -> None:
    conn = connect()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO run_origin (run_id, benchmark, target, note, origin_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, benchmark, target, note, json.dumps(origin)),
        )
    conn.close()


def get_origin(run_id: str) -> Optional[sqlite3.Row]:
    conn = connect()
    row = conn.execute("SELECT * FROM run_origin WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    return row


def set_metrics(run_id: str, metrics: dict[str, Any]) -> None:
    conn = connect()
    with conn:
        conn.execute(
            "UPDATE run_origin SET metrics_json = ? WHERE run_id = ?",
            (json.dumps(metrics), run_id),
        )
    conn.close()


# --- shelf / artifacts ---


def insert_artifact(
    artifact_id: str,
    benchmark: str,
    layer: str,
    dataset_sha: Optional[str],
    path: str,
    origin: dict[str, Any],
    created_by_run: Optional[str] = None,
) -> None:
    conn = connect()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO artifacts "
            "(id, benchmark, layer, dataset_sha, path, created_by_run, origin_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (artifact_id, benchmark, layer, dataset_sha, path, created_by_run, json.dumps(origin)),
        )
    conn.close()


def get_artifact(artifact_id: str) -> Optional[sqlite3.Row]:
    conn = connect()
    row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
    conn.close()
    return row


def list_artifacts(
    benchmark: Optional[str] = None, layer: Optional[str] = None, limit: int = 50
) -> list[sqlite3.Row]:
    conn = connect()
    sql = "SELECT * FROM artifacts WHERE 1=1"
    params: list[Any] = []
    if benchmark:
        sql += " AND benchmark = ?"
        params.append(benchmark)
    if layer:
        sql += " AND layer = ?"
        params.append(layer)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows
