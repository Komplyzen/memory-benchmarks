"""
No-Memory Client (full-context floor)
=====================================

``--backend none``: no memory system at all. ``add`` keeps the raw transcript
per ``user_id`` in process memory; ``search`` returns the whole transcript as a
single memory, so the answerer sees the full conversation plus the question.
This is the floor the small-model KG legs are compared against.

LoCoMo conversations fit (about 9k tokens). LongMemEval histories may not, so
the transcript is truncated from the oldest session until it fits a token
budget (chars / 4 approximation) and the drop is recorded in ``query_debug``
and in run metadata.

Runners use a single pseudo-cutoff ``full_context`` for this backend (see
``benchmarks.common.utils.FULL_CONTEXT_CUTOFF``).
"""

from __future__ import annotations

import logging
from typing import Any

from benchmarks.common.kg_agent_ingest import render_session

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 150_000
CHARS_PER_TOKEN = 4


class NoMemoryClient:
    """Same surface as Mem0Client; stores transcripts, retrieves them whole."""

    def __init__(self, context_tokens: int = DEFAULT_CONTEXT_TOKENS, chars_per_token: int = CHARS_PER_TOKEN):
        self.context_tokens = int(context_tokens)
        self.chars_per_token = chars_per_token
        self._sessions: dict[str, dict[Any, dict[str, Any]]] = {}  # user_id -> key -> {"date", "order", "lines"}
        self._order: dict[str, int] = {}
        self.stats = {"questions": 0, "questions_truncated": 0, "sessions_dropped_total": 0, "chars_dropped_total": 0}

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> NoMemoryClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        key = timestamp if timestamp is not None else (observation_date or "undated")
        sessions = self._sessions.setdefault(user_id, {})
        if key not in sessions:
            self._order[user_id] = self._order.get(user_id, 0) + 1
            date = observation_date
            if date is None and timestamp is not None:
                from datetime import datetime, timezone
                try:
                    date = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")
                except (OverflowError, OSError, ValueError):
                    date = None
            sessions[key] = {"date": date or "unknown date", "order": self._order[user_id], "lines": []}
        sessions[key]["lines"].append(render_session(messages))
        return {"results": []}

    def _transcript(self, user_id: str) -> tuple[str, dict[str, Any]]:
        sessions = sorted(self._sessions.get(user_id, {}).values(), key=lambda s: s["order"])
        blocks = [f"=== Session dated {s['date']} ===\n" + "\n".join(s["lines"]) for s in sessions]
        budget_chars = self.context_tokens * self.chars_per_token
        total_chars = sum(len(b) + 2 for b in blocks)
        dropped = 0
        dropped_chars = 0
        while len(blocks) > 1 and sum(len(b) + 2 for b in blocks) > budget_chars:
            dropped_chars += len(blocks[0]) + 2
            blocks.pop(0)
            dropped += 1
        text = "\n\n".join(blocks)
        debug = {
            "backend": "none",
            "sessions_total": len(sessions),
            "sessions_dropped_oldest": dropped,
            "chars_total": total_chars,
            "chars_kept": len(text),
            "chars_dropped": dropped_chars,
            "approx_tokens_kept": len(text) // self.chars_per_token,
            "context_token_budget": self.context_tokens,
        }
        return text, debug

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> dict[str, Any] | list[dict]:
        text, debug = self._transcript(user_id)
        self.stats["questions"] += 1
        if debug["sessions_dropped_oldest"]:
            self.stats["questions_truncated"] += 1
            self.stats["sessions_dropped_total"] += debug["sessions_dropped_oldest"]
            self.stats["chars_dropped_total"] += debug["chars_dropped"]
        if not text:
            return {"results": [], "query_debug": debug}
        return {
            "results": [{"memory": text, "score": 1.0, "id": f"full_context:{user_id}"}],
            "query_debug": debug,
        }

    async def get_user_profile(self, user_id: str) -> dict | None:
        return None

    async def delete_user(self, user_id: str) -> bool:
        self._sessions.pop(user_id, None)
        self._order.pop(user_id, None)
        return True

    def ingest_metadata(self) -> dict[str, Any]:
        return {
            "memory_backend": "none",
            "none_context_token_budget": self.context_tokens,
            "none_questions": self.stats["questions"],
            "none_questions_truncated": self.stats["questions_truncated"],
            "none_sessions_dropped_total": self.stats["sessions_dropped_total"],
            "none_chars_dropped_total": self.stats["chars_dropped_total"],
        }
