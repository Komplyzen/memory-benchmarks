"""Bounded concurrency for independent jobs (one job per benchmark question).

One job: run coroutines with at most ``workers`` in flight and never leave a task running after
returning. When a job fails, jobs that have not started yet are skipped (so a systematic error,
such as a bad API key, does not burn through the remaining work, matching a plain ``for`` loop that
stops at the first exception), jobs already in flight run to completion (their results are usually
saved to disk as they finish), and the first failure is re-raised once everything has settled.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Sequence, TypeVar

T = TypeVar("T")


async def gather_bounded(jobs: Sequence[Callable[[], Awaitable[T]]], workers: int) -> list[T]:
    """Run ``jobs`` (zero-argument coroutine functions) with at most ``workers`` in flight.

    Results are returned in job order. With ``workers == 1`` jobs run strictly one after another,
    in order, exactly like a plain ``for`` loop.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    sem = asyncio.Semaphore(workers)
    failed = False

    async def run(job: Callable[[], Awaitable[T]]) -> T | None:
        nonlocal failed
        async with sem:
            if failed:
                return None  # skipped: an earlier job already failed
            try:
                return await job()
            except BaseException:
                failed = True
                raise

    results = await asyncio.gather(*(run(j) for j in jobs), return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return list(results)  # type: ignore[arg-type]
