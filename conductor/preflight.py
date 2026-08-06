"""Fail-loud preflight: verify the memory platform + embedder actually work
before any run/materialize. NEVER falls back to a stand-in -- the embedder is the
benchmarked component; if it's down the run must not proceed. On failure we halt
loudly and record nothing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class PreflightError(RuntimeError):
    """Raised when the target platform/embedder is not usable."""


@dataclass
class PreflightResult:
    ok: bool
    host: str
    summary: str
    hint: Optional[str] = None


_OSS_DEFAULT = "http://localhost:8888"
_CLOUD_DEFAULT = "https://api.mem0.ai"


def resolve_host(config: dict[str, Any], env_overrides: dict[str, str]) -> str:
    """Mirror benchmarks/common/mem0_client host resolution."""
    backend = config.get("backend", "oss")
    default = _CLOUD_DEFAULT if backend == "cloud" else _OSS_DEFAULT
    host = (
        config.get("mem0_host")
        or env_overrides.get("MEM0_HOST")
        or os.environ.get("MEM0_HOST")
        or default
    )
    return str(host).rstrip("/")


def _classify(status: Optional[int], body: str) -> tuple[str, str]:
    """(summary, hint) from an error body."""
    b = body.lower()
    if any(k in b for k in ("security token", "unrecognizedclient", "expiredtoken", "invalid the security")):
        return ("embedder credentials rejected (expired/invalid)",
                "Refresh the AWS/SSO creds in .env (AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN), then restart the mem0 container.")
    if "accessdenied" in b or "not authorized" in b:
        return ("embedder access denied (IAM)",
                "The AWS identity lacks sagemaker:InvokeEndpoint on the embedding endpoint.")
    if "not found" in b or "validationerror" in b or "endpoint" in b:
        return ("embedder endpoint not found",
                "Check SAGEMAKER_ENDPOINT_NAME matches a deployed InService endpoint in this account/region.")
    return (f"platform error (HTTP {status})", body[:200])


def probe(host: str, backend: str = "oss", timeout: float = 20.0) -> PreflightResult:
    """Health + a non-mutating embedder exercise (a search embeds its query)."""
    # 1. reachability / health
    health_url = f"{host}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as r:
            if r.status != 200:
                return PreflightResult(False, host, f"health returned HTTP {r.status}")
    except urllib.error.HTTPError as e:
        return PreflightResult(False, host, f"health HTTP {e.code}", None)
    except Exception:
        return PreflightResult(False, host, f"platform unreachable at {host}",
                               "Is the mem0 server / sandbox running? (docker compose up)")

    # 2. embedder exercise: a search embeds the query but writes nothing.
    search_url = f"{host}/search"
    payload = json.dumps({"query": "conductor preflight probe",
                          "user_id": "__conductor_preflight__", "limit": 1}).encode()
    req = urllib.request.Request(search_url, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return PreflightResult(True, host, "platform healthy; embedder responding")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        summary, hint = _classify(e.code, body)
        return PreflightResult(False, host, summary, hint)
    except Exception as e:
        return PreflightResult(False, host, f"embedder probe failed: {type(e).__name__}", str(e)[:200])


def probe_v3(host: str, *, mode: str, user_id: str, api_key: str, timeout: float = 120.0) -> PreflightResult:
    paths = {
        "regular": "/v3/memories/search/",
        "fast": "/v3/memories/search/fast/",
        "agentic": "/v3/memories/search/agentic/",
    }
    if mode not in paths:
        return PreflightResult(False, host, f"unknown v3 search mode {mode!r}")
    payload: dict[str, Any] = {
        "query": "conductor BrowseComp preflight probe",
        "filters": {"user_id": user_id},
        "top_k": 1,
    }
    if mode in {"regular", "fast"}:
        payload["threshold"] = 0.0
    if mode == "regular":
        payload["rerank"] = False
    request = urllib.request.Request(
        host.rstrip("/") + paths[mode],
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
        return PreflightResult(True, host, f"v3 {mode} search responding")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return PreflightResult(False, host, f"v3 {mode} returned HTTP {exc.code}", body[:300])
    except Exception as exc:
        return PreflightResult(False, host, f"v3 {mode} probe failed: {type(exc).__name__}", str(exc)[:200])


def require_ok(config: dict[str, Any], env_overrides: dict[str, str]) -> str:
    """Probe and raise PreflightError (loud) on any failure. Returns the host."""
    host = resolve_host(config, env_overrides)
    if config.get("store_manifest") and config.get("search_mode"):
        manifest = json.loads(Path(config["store_manifest"]).read_text(encoding="utf-8"))
        api_key = env_overrides.get("MEM0_API_KEY") or os.environ.get("MEM0_API_KEY")
        if not api_key:
            raise PreflightError("PREFLIGHT FAILED: MEM0_API_KEY is required for BrowseComp platform runs")
        res = probe_v3(
            host,
            mode=str(config["search_mode"]),
            user_id=manifest["scope"]["user_id"],
            api_key=api_key,
        )
    else:
        res = probe(host, backend=config.get("backend", "oss"))
    if not res.ok:
        msg = f"PREFLIGHT FAILED @ {res.host}: {res.summary}"
        if res.hint:
            msg += f"\n  fix: {res.hint}"
        msg += "\n  (not falling back -- the embedder is the benchmarked component; nothing was recorded.)"
        raise PreflightError(msg)
    return host
