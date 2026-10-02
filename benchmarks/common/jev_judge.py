"""Jev (TypeSafe System One) as a LoCoMo answer judge.

One job: turn (question, gold answer, generated answer) into a CORRECT/WRONG verdict plus the
probability behind it. It does not read result files and does not compute metrics; see
scripts/compare_judges.py for that.

Jev is not a chat model. It takes a ``state`` and typed ``questions`` and returns probabilities,
with no explanation. The judge rules from benchmarks/locomo/prompts.py are therefore restated as
the Noul question's instructions and criteria (JEV_RULES below). That restatement is hand-condensed
and is the main fidelity risk of this judge: review it before trusting a comparison.

API: POST https://api.typesafe.ai/v1/systemone, ``Authorization: Bearer $TYPESAFE_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import aiohttp

from benchmarks.locomo.prompts import preprocess_answer

logger = logging.getLogger(__name__)

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
QUESTION_KEY = "answer_is_correct"

# Condensed from _JUDGE_TEMPLATE in benchmarks/locomo/prompts.py (rules 1-7 and the WRONG clause).
JEV_INSTRUCTIONS = (
    "Is the generated answer CORRECT with respect to the gold answer for this question? "
    "Judge whether the system recalled the right fact, not the wording."
)
JEV_CRITERIA_TRUE = (
    "The generated answer contains at least one correct item from the gold answer (partial credit "
    "counts), or expresses the same concept in different words, or identifies the same named "
    "entity, or adds extra detail on top of the gold facts. Dates within 14 days, and durations "
    "within 50%, match; a relative date matches a specific date in the same window. Emotions in the "
    "same positive or negative family about the same event match."
)
JEV_CRITERIA_FALSE = (
    "The generated answer contains none of the gold answer's items, addresses a different topic, "
    "or shows a genuinely different or incorrect understanding of the fact."
)


@dataclass(frozen=True)
class JevVerdict:
    correct: bool
    p_yes: float
    input_tokens: int
    output_tokens: int


class JevError(RuntimeError):
    """The API gave no usable answer after retries (status, request id and body in the message)."""


def build_request(category: int, question: str, gold: str, generated: str) -> dict[str, Any]:
    """The request body for one verdict. Pure, so --dry-run can print it with no API key."""
    return {
        "model": JEV_MODEL,
        "state": {
            "question": question,
            "gold_answer": preprocess_answer(category, str(gold)),
            "generated_answer": generated,
        },
        "questions": {
            QUESTION_KEY: {
                "type": "noul",
                "instructions": JEV_INSTRUCTIONS,
                "criteria": {"true": JEV_CRITERIA_TRUE, "false": JEV_CRITERIA_FALSE},
            }
        },
    }


def parse_response(body: dict[str, Any], threshold: float = 0.5) -> JevVerdict:
    """Extract the Noul probability. Raises JevError when the expected answer is missing."""
    try:
        p_yes = float(body["answers"][QUESTION_KEY]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"unexpected Jev response shape: {json.dumps(body)[:300]}") from exc
    usage = body.get("usage") or {}
    return JevVerdict(
        correct=p_yes > threshold,
        p_yes=p_yes,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
    )


class JevJudge:
    """Async Jev client for one judging role. Use as ``async with JevJudge(...) as judge``."""

    def __init__(self, api_key: str | None = None, *, threshold: float = 0.5,
                 concurrency: int = 8, max_retries: int = 5, timeout_s: float = 30.0) -> None:
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self._api_key:
            raise JevError("TYPESAFE_API_KEY is not set")
        self._threshold = threshold
        self._sem = asyncio.Semaphore(concurrency)
        self._max_retries = max_retries
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "JevJudge":
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._session is not None:
            await self._session.close()

    async def judge(self, category: int, question: str, gold: str, generated: str) -> JevVerdict:
        assert self._session is not None, "use JevJudge as an async context manager"
        payload = build_request(category, question, gold, generated)
        last = ""
        async with self._sem:
            for attempt in range(self._max_retries):
                async with self._session.post(JEV_URL, json=payload) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        return parse_response(json.loads(text), self._threshold)
                    last = f"HTTP {resp.status}: {text[:200]}"
                    if resp.status not in (429, 529) and resp.status < 500:
                        raise JevError(last)  # a 4xx other than rate limiting will not fix itself
                logger.warning("Jev attempt %d/%d failed: %s", attempt + 1, self._max_retries, last)
                await asyncio.sleep(2 ** attempt)
        raise JevError(f"gave up after {self._max_retries} attempts; last: {last}")
