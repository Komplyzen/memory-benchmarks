"""The detached reaper for one run.

Spawned by launcher.start() in its own session. It owns the child pipeline
process's lifecycle: writes the log, flips ledger status, and on exit records the
result-file path the pipeline printed. Because it is the session/group leader, a
`conductor stop` can kill the whole group by this process's pid.

Status transitions mirror src/lib/executor.ts finish(): a run already marked
`stopped` (by `conductor stop`) is never overwritten with succeeded/failed.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from . import db
from .paths import run_dir

_RESULT_RE = re.compile(r"Results saved to:\s*(.+)")


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _child_env(env_overrides: dict[str, str]) -> dict[str, str]:
    """Mirror executor.ts buildScriptEnv: drop inherited MEM0_* so the repo's .env
    wins, then apply explicit overrides."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEM0_")}
    # Ensure `-m benchmarks.*` resolves even if the child changes cwd.
    repo = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo + (os.pathsep + existing if existing else "")
    env.update(env_overrides)
    return env


def _parse_result_file(log_path: Path, cwd: Path) -> str | None:
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None
    matches = _RESULT_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    p = Path(raw)
    return str(p if p.is_absolute() else cwd / p)


def run(run_id: str) -> int:
    spec = json.loads((run_dir(run_id) / "launch.json").read_text())
    argv = spec["argv"]
    cwd = Path(spec["cwd"])
    log_path = Path(spec["log_file"])
    env_overrides = spec.get("env_overrides", {})

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", buffering=1)
    log.write(
        f"[conductor] run {run_id}\n"
        f"argv: {' '.join(argv)}\n"
        f"time: {_utc_now_iso()}\n" + "=" * 60 + "\n\n"
    )
    log.flush()

    # Our pid is the session/group leader (launcher used start_new_session=True), so
    # storing it lets `stop` kill the child too via killpg.
    db.update_run(run_id, status="running", pid=os.getpid(), started_at=_utc_now_iso())

    child = subprocess.Popen(argv, cwd=str(cwd), env=_child_env(env_overrides),
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)

    def _forward(signum, _frame):
        try:
            child.terminate()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    code = child.wait()

    log.write(f"\n{'=' * 60}\n[conductor] finished with code {code}\n")
    log.flush()
    log.close()

    # Don't clobber a deliberate stop.
    current = db.get_run(run_id)
    if current is not None and current["status"] == "stopped":
        return code

    result_file = _parse_result_file(log_path, cwd)
    fields = {"status": "succeeded" if code == 0 else "failed", "finished_at": _utc_now_iso()}
    if result_file:
        fields["result_file"] = result_file
    db.update_run(run_id, **fields)

    # Lift metrics into the ledger so `ls`/`diff` can answer "what did we get".
    if code == 0 and result_file:
        try:
            from . import metrics as metrics_mod
            row = db.get_run(run_id)
            benchmark = row["template_id"] if row else None
            m = metrics_mod.extract(benchmark, result_file) if benchmark else None
            if m:
                db.set_metrics(run_id, m)
        except Exception:
            pass  # metrics are a convenience; never fail the run over them
    return code


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m conductor.supervisor <run_id>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run(sys.argv[1]))


if __name__ == "__main__":
    main()
