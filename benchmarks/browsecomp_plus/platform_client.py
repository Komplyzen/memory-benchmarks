from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


_PATHS = {
    "regular": "/v3/memories/search/",
    "fast": "/v3/memories/search/fast/",
    "agentic": "/v3/memories/search/agentic/",
}


@dataclass(frozen=True)
class SearchResponse:
    body: dict
    wall_ms: float
    server_timing: str | None


class PlatformClient:
    def __init__(self, host: str, api_key: str, user_id: str, timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self.timeout = timeout

    def search(self, *, mode: str, query: str, top_k: int) -> SearchResponse:
        if mode not in _PATHS:
            raise ValueError(f"Unknown search mode: {mode}")
        payload = {
            "query": query,
            "filters": {"user_id": self.user_id},
            "top_k": top_k,
        }
        if mode in {"regular", "fast"}:
            payload["threshold"] = 0.0
        if mode == "regular":
            payload["rerank"] = False
        request = urllib.request.Request(
            self.host + _PATHS[mode],
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
                return SearchResponse(
                    body=body,
                    wall_ms=round((time.perf_counter() - started) * 1000, 3),
                    server_timing=response.headers.get("Server-Timing"),
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"{mode} search returned HTTP {exc.code}: {detail[:500]}") from exc
