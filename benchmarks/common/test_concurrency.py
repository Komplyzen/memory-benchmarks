"""Run: python -m unittest benchmarks.common.test_concurrency"""

import asyncio
import unittest

from benchmarks.common.concurrency import gather_bounded


class GatherBoundedTest(unittest.TestCase):
    def test_workers_bound_the_number_in_flight(self) -> None:
        live = peak = 0

        async def job(i: int) -> int:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return i

        out = asyncio.run(gather_bounded([lambda i=i: job(i) for i in range(12)], 3))
        self.assertEqual(out, list(range(12)))  # results in job order
        self.assertEqual(peak, 3)

    def test_one_worker_is_strictly_sequential_and_ordered(self) -> None:
        order: list[int] = []

        async def job(i: int) -> None:
            order.append(i)
            await asyncio.sleep(0)
            order.append(-i)

        asyncio.run(gather_bounded([lambda i=i: job(i) for i in range(1, 5)], 1))
        self.assertEqual(order, [1, -1, 2, -2, 3, -3, 4, -4])

    def test_failure_with_one_worker_stops_later_jobs_like_a_for_loop(self) -> None:
        ran: list[int] = []

        async def job(i: int) -> None:
            ran.append(i)
            if i == 2:
                raise RuntimeError("bad key")

        with self.assertRaises(RuntimeError):
            asyncio.run(gather_bounded([lambda i=i: job(i) for i in range(6)], 1))
        self.assertEqual(ran, [0, 1, 2])

    def test_failure_lets_in_flight_jobs_finish_and_nothing_outlives_the_call(self) -> None:
        finished: list[int] = []

        async def job(i: int) -> None:
            if i == 0:
                await asyncio.sleep(0.01)  # fail while jobs 1 and 2 are already in flight
                raise RuntimeError("boom")
            await asyncio.sleep(0.05)
            finished.append(i)

        with self.assertRaises(RuntimeError):
            asyncio.run(gather_bounded([lambda i=i: job(i) for i in range(3)], 3))
        self.assertEqual(sorted(finished), [1, 2])  # in-flight siblings completed before the raise

    def test_empty_and_invalid_workers(self) -> None:
        self.assertEqual(asyncio.run(gather_bounded([], 4)), [])
        with self.assertRaises(ValueError):
            asyncio.run(gather_bounded([], 0))


if __name__ == "__main__":
    unittest.main()
