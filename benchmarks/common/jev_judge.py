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
import math
import os
import random
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
    "within 50%, match; a relative date matches a specific date in the same window; a specific date "
    "consistent with a vague reference (for example February 2020 for 'a few years ago' relative to "
    "2023) matches; converting 'last year' to the actual year matches. Emotions in the same positive "
    "or negative family about the same event match."
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


class JevCircuitOpen(JevError):
    """Too many consecutive failed judgments; further calls are refused without hitting the API."""


# Statuses worth retrying: timeouts, rate limits, overload, server errors. Any other 4xx is permanent.
RETRYABLE_STATUS = frozenset({408, 425, 429, 529})
MAX_BACKOFF_S = 60.0


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


def _tokens(usage: Any, key: str) -> int:
    """Token counts are informational: anything that is not a non-negative int counts as 0."""
    v = usage.get(key) if isinstance(usage, dict) else None
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def parse_response(body: Any, threshold: float = 0.5) -> JevVerdict:
    """Extract the Noul probability. Raises JevError for any body that is not a usable verdict."""
    try:
        p_yes = float(body["answers"][QUESTION_KEY]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"unexpected Jev response shape: {json.dumps(body, default=str)[:300]}") from exc
    if not math.isfinite(p_yes) or not 0.0 <= p_yes <= 1.0:
        raise JevError(f"Jev probability out of range: {p_yes!r}")
    usage = body.get("usage")
    return JevVerdict(
        correct=p_yes > threshold,
        p_yes=p_yes,
        input_tokens=_tokens(usage, "input_tokens"),
        output_tokens=_tokens(usage, "output_tokens"),
    )


def backoff_delay(attempt: int, retry_after: str | None = None, rng: random.Random | None = None) -> float:
    """Seconds to wait before retry ``attempt + 1``: Retry-After when given, else full jitter on 2**attempt."""
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), MAX_BACKOFF_S)
        except ValueError:
            pass  # an HTTP-date Retry-After is not supported; fall back to jitter
    return (rng or random).uniform(0.0, min(2.0 ** attempt, MAX_BACKOFF_S))


class JevJudge:
    """Async Jev client for one judging role. Use as ``async with JevJudge(...) as judge``.

    A request that times out may still have been processed and billed, so retries after a timeout
    can pay twice; ``breaker`` bounds the damage by refusing calls after that many consecutive
    judgments that exhausted their retries (or hit a permanent error such as a bad key).
    """

    def __init__(self, api_key: str | None = None, *, threshold: float = 0.5,
                 concurrency: int = 8, max_retries: int = 5, timeout_s: float = 30.0,
                 breaker: int = 10) -> None:
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self._api_key:
            raise JevError("TYPESAFE_API_KEY is not set")
        self._threshold = threshold
        self._sem = asyncio.Semaphore(concurrency)
        self._max_retries = max_retries
        self._breaker = breaker
        self._consecutive_failures = 0
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
        if self._consecutive_failures >= self._breaker:
            raise JevCircuitOpen(f"{self._consecutive_failures} consecutive judgments failed; not calling Jev")
        try:
            verdict = await self._judge(category, question, gold, generated)
        except JevError:
            self._consecutive_failures += 1
            raise
        self._consecutive_failures = 0
        return verdict

    async def _judge(self, category: int, question: str, gold: str, generated: str) -> JevVerdict:
        assert self._session is not None, "use JevJudge as an async context manager"
        payload = build_request(category, question, gold, generated)
        last = ""
        for attempt in range(self._max_retries):
            retry_after: str | None = None
            # The semaphore bounds in-flight requests only; backoff sleeps happen outside it.
            async with self._sem:
                try:
                    async with self._session.post(JEV_URL, json=payload) as resp:
                        status = resp.status
                        retry_after = (getattr(resp, "headers", None) or {}).get("Retry-After")
                        text = await resp.text()
                except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeDecodeError) as exc:
                    status, text = None, f"{type(exc).__name__}: {exc}"
            if status == 200:
                try:
                    return parse_response(json.loads(text), self._threshold)
                except ValueError:  # json.loads failure; a JevError from parse_response is permanent
                    last = f"HTTP 200 with non-JSON body: {text[:200]}"
            else:
                last = f"HTTP {status}: {text[:200]}" if status is not None else text
                if status is not None and status not in RETRYABLE_STATUS and status < 500:
                    raise JevError(last)  # a 4xx other than the retryable ones will not fix itself
            logger.warning("Jev attempt %d/%d failed: %s", attempt + 1, self._max_retries, last)
            if attempt + 1 < self._max_retries:
                await asyncio.sleep(backoff_delay(attempt, retry_after))
        raise JevError(f"gave up after {self._max_retries} attempts; last: {last}")
