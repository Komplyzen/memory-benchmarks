"""Tests for the Jev judge comparison. Run: python -m unittest scripts.test_compare_judges

The Jev HTTP call is the only thing faked (an injected judge function); loading real result files,
verdict parsing and the statistics all run for real.
"""

import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aiohttp

from benchmarks.common import jev_judge
from benchmarks.common.jev_judge import (
    JEV_CRITERIA_TRUE, JevError, JevJudge, JevVerdict, build_request, parse_response, QUESTION_KEY,
)
from scripts import compare_judges
from scripts.compare_judges import cohens_kappa, judge_all, load_items, summarize


def _write(dirpath: Path, qid: str, cat: int, gold: str, cutoffs: dict[str, tuple[str, str]],
           reason: str = "r") -> None:
    (dirpath / f"{qid}.json").write_text(json.dumps({
        "question_id": qid, "category": cat, "category_name": "single-hop", "question": f"q {qid}?",
        "ground_truth_answer": gold,
        "cutoff_results": {k: {"judgment": j, "generated_answer": g, "reason": reason} for k, (j, g) in cutoffs.items()},
    }))


class _Resp:
    def __init__(self, status: int, text: str) -> None:
        self.status, self._text = status, text

    async def text(self) -> str:
        return self._text

    async def __aenter__(self) -> "_Resp":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeSession:
    """Plays back scripted outcomes: a (status, text) tuple, or an exception to raise from post()."""
    def __init__(self, outcomes: list) -> None:
        self.outcomes, self.calls = list(outcomes), 0

    def post(self, url: str, json: dict) -> _Resp:
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return _Resp(*o)


OK_BODY = json.dumps({"answers": {QUESTION_KEY: {"noul": 0.9}}, "usage": {"input_tokens": 3, "output_tokens": 1}})


def _judge_with(outcomes: list, retries: int = 3) -> tuple[JevJudge, FakeSession]:
    j = JevJudge(api_key="k", max_retries=retries)
    j._session = FakeSession(outcomes)  # type: ignore[assignment]
    return j, j._session  # type: ignore[return-value]


class FakeJev:
    """Says yes iff the generated answer contains the word 'good'. Fails on 'boom'."""
    async def __call__(self, category: int, question: str, gold: str, generated: str) -> JevVerdict:
        if "boom" in generated:
            raise JevError("HTTP 500")
        p = 0.9 if "good" in generated else 0.1
        return JevVerdict(correct=p > 0.5, p_yes=p, input_tokens=10, output_tokens=1)


class CompareJudgesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        _write(self.dir, "conv0_q0", 1, "a", {"top_20": ("CORRECT", "good"), "top_100": ("WRONG", "bad")})
        _write(self.dir, "conv0_q1", 3, "x; y", {"top_20": ("CORRECT", "bad")})      # disagreement
        _write(self.dir, "conv0_q2", 1, "b", {"top_20": ("WRONG", "boom")})          # Jev error
        (self.dir / "conv0_q3.json").write_text(json.dumps({"question_id": "conv0_q3", "category": 1}))  # no cutoffs

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_load_uses_stored_generated_answers_and_skips_unjudged(self) -> None:
        items = load_items(self.dir)
        self.assertEqual(len(items), 4)
        self.assertEqual({i["question_id"] for i in items}, {"conv0_q0", "conv0_q1", "conv0_q2"})
        self.assertEqual(load_items(self.dir, limit=2).__len__(), 2)

    def test_summary_counts_agreement_confusion_and_errors(self) -> None:
        rows = asyncio.run(judge_all(load_items(self.dir), FakeJev()))
        s = summarize(rows)
        self.assertEqual((s["judged"], s["errors"]), (3, 1))
        self.assertEqual(s["agreed"], 2)
        self.assertAlmostEqual(s["agreement"], 2 / 3)
        self.assertEqual(s["confusion"]["original_correct_jev_wrong"], 1)
        self.assertEqual([d["question_id"] for d in s["disagreements"]], ["conv0_q1"])
        self.assertEqual(s["by_cutoff"]["top_20"]["n"], 2)

    def test_kappa_perfect_chance_and_degenerate(self) -> None:
        self.assertEqual(cohens_kappa([True, False], [True, False]), 1.0)
        self.assertEqual(cohens_kappa([True, False], [False, True]), -1.0)
        self.assertIsNone(cohens_kappa([True, True], [True, True]))
        self.assertIsNone(cohens_kappa([], []))

    def test_request_applies_category3_gold_preprocessing(self) -> None:
        req = build_request(3, "q?", "first; second", "gen")
        self.assertEqual(req["state"]["gold_answer"], "first")
        self.assertEqual(build_request(1, "q?", "first; second", "gen")["state"]["gold_answer"], "first; second")
        self.assertEqual(req["questions"][QUESTION_KEY]["type"], "noul")

    def test_parse_response_threshold_and_bad_shape(self) -> None:
        body = {"answers": {QUESTION_KEY: {"type": "noul", "noul": 0.6}}, "usage": {"input_tokens": 5, "output_tokens": 2}}
        self.assertTrue(parse_response(body).correct)
        self.assertFalse(parse_response(body, threshold=0.7).correct)
        with self.assertRaises(JevError):
            parse_response({"answers": {}})


class JevJudgeRetryTest(unittest.TestCase):
    def run_judge(self, outcomes: list, retries: int = 3):
        j, sess = _judge_with(outcomes, retries)
        with mock.patch.object(jev_judge.asyncio, "sleep", new=mock.AsyncMock()):
            return asyncio.run(j.judge(1, "q", "g", "a")), sess

    def test_network_error_is_retried_then_succeeds(self) -> None:
        v, sess = self.run_judge([aiohttp.ClientConnectionError("reset"), asyncio.TimeoutError(), (200, OK_BODY)])
        self.assertTrue(v.correct)
        self.assertEqual(sess.calls, 3)

    def test_persistent_network_error_becomes_jev_error_not_a_crash(self) -> None:
        with self.assertRaises(JevError) as cm:
            self.run_judge([asyncio.TimeoutError()] * 3)
        self.assertIn("gave up after 3", str(cm.exception))

    def test_non_json_200_is_retried_and_then_jev_error(self) -> None:
        v, sess = self.run_judge([(200, "<html>"), (200, OK_BODY)])
        self.assertTrue(v.correct)
        with self.assertRaises(JevError):
            self.run_judge([(200, "<html>")] * 3)

    def test_rate_limit_and_5xx_retry_but_other_4xx_does_not(self) -> None:
        v, sess = self.run_judge([(429, "slow"), (503, "down"), (200, OK_BODY)])
        self.assertEqual(sess.calls, 3)
        j, sess = _judge_with([(401, "no"), (200, OK_BODY)])
        with mock.patch.object(jev_judge.asyncio, "sleep", new=mock.AsyncMock()):
            with self.assertRaises(JevError):
                asyncio.run(j.judge(1, "q", "g", "a"))
        self.assertEqual(sess.calls, 1)

    def test_criteria_keep_the_date_rules_from_the_original_prompt(self) -> None:
        for phrase in ("14 days", "50%", "vague reference", "'last year'"):
            self.assertIn(phrase, JEV_CRITERIA_TRUE)


class CompareRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pred = self.root / "predicted_x"
        self.pred.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_original_judge_failure_is_skipped_not_compared(self) -> None:
        _write(self.pred, "conv0_q0", 1, "a", {"top_20": ("WRONG", "good")}, reason="")  # judge call failed
        _write(self.pred, "conv0_q1", 1, "a", {"top_20": ("CORRECT", "good")})
        rows = asyncio.run(judge_all(load_items(self.pred), FakeJev()))
        s = summarize(rows)
        self.assertEqual((s["judged"], s["skipped_original_failed"], s["errors"]), (1, 1, 0))
        self.assertEqual(s["agreement"], 1.0)

    def test_limit_spreads_over_categories_and_cutoffs(self) -> None:
        for i in range(6):
            _write(self.pred, f"conv0_q{i}", 1, "a", {"top_20": ("CORRECT", "good")})
        _write(self.pred, "conv1_q0", 2, "a", {"top_20": ("CORRECT", "good"), "top_100": ("CORRECT", "good")})
        got = load_items(self.pred, limit=3)
        self.assertEqual({(i["category"], i["cutoff"]) for i in got}, {(1, "top_20"), (2, "top_20"), (2, "top_100")})
        self.assertEqual(load_items(self.pred, limit=3), got)  # deterministic

    def test_on_row_sees_every_row_as_it_finishes(self) -> None:
        _write(self.pred, "conv0_q0", 1, "a", {"top_20": ("CORRECT", "good"), "top_100": ("WRONG", "boom")})
        seen: list[dict] = []
        asyncio.run(judge_all(load_items(self.pred), FakeJev(), seen.append))
        self.assertEqual(len(seen), 2)

    def test_run_writes_report_and_progress_outside_the_result_dir(self) -> None:
        _write(self.pred, "conv0_q0", 1, "a", {"top_20": ("CORRECT", "good")})

        class Ctx:
            def __init__(self, **kw: object) -> None: ...
            async def __aenter__(self) -> "Ctx": return self
            async def __aexit__(self, *e: object) -> None: ...
            judge = FakeJev()

        args = argparse.Namespace(predicted_dir=str(self.pred), out=None, threshold=0.5, limit=None,
                                  concurrency=2, dry_run=False)
        with mock.patch.object(compare_judges, "JevJudge", Ctx):
            self.assertEqual(asyncio.run(compare_judges.run(args)), 0)
        self.assertEqual(list(self.pred.glob("judge_compare*")), [])  # result dir untouched
        out = self.root / "judge_compare"
        self.assertEqual(len(list(out.glob("*.rows.jsonl"))), 1)
        self.assertEqual(len(list(out.glob("*.md"))), 1)
        self.assertIn("p>0.3", next(out.glob("*.md")).read_text())


if __name__ == "__main__":
    unittest.main()
