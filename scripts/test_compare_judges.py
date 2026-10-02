"""Tests for the Jev judge comparison. Run: python -m unittest scripts.test_compare_judges

The Jev HTTP call is the only thing faked (an injected judge function, or a scripted session
standing in for aiohttp); loading real result files, verdict parsing, retries, resume and the
statistics all run for real.
"""

import argparse
import asyncio
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aiohttp

from benchmarks.common import jev_judge
from benchmarks.common.jev_judge import (
    JEV_CRITERIA_TRUE, JevCircuitOpen, JevError, JevJudge, JevVerdict, QUESTION_KEY, backoff_delay,
    build_request, parse_response,
)
from scripts import compare_judges
from scripts.compare_judges import (
    apply_threshold, cohens_kappa, judge_all, load_items, load_progress, run_id, select_items, summarize,
)


def _write(dirpath: Path, qid: str, cat: int, gold: str, cutoffs: dict[str, tuple[str, str]],
           reason: str = "r", judge_failed: bool | None = None) -> None:
    def cut(j: str, g: str) -> dict:
        c = {"judgment": j, "generated_answer": g, "reason": reason}
        if judge_failed is not None:
            c["judge_failed"] = judge_failed
        return c
    (dirpath / f"{qid}.json").write_text(json.dumps({
        "question_id": qid, "category": cat, "category_name": "single-hop", "question": f"q {qid}?",
        "ground_truth_answer": gold,
        "cutoff_results": {k: cut(j, g) for k, (j, g) in cutoffs.items()},
    }))


class _Resp:
    def __init__(self, status: int, text: str | Exception, headers: dict | None = None) -> None:
        self.status, self._text, self.headers = status, text, headers or {}

    async def text(self) -> str:
        if isinstance(self._text, Exception):
            raise self._text
        return self._text

    async def __aenter__(self) -> "_Resp":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeSession:
    """Plays back scripted outcomes: a (status, text[, headers]) tuple, or an exception to raise from post()."""
    def __init__(self, outcomes: list) -> None:
        self.outcomes, self.calls = list(outcomes), 0

    def post(self, url: str, json: dict) -> _Resp:
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return _Resp(*o)


OK_BODY = json.dumps({"answers": {QUESTION_KEY: {"noul": 0.9}}, "usage": {"input_tokens": 3, "output_tokens": 1}})


def _judge_with(outcomes: list, retries: int = 3, **kw) -> tuple[JevJudge, FakeSession]:
    j = JevJudge(api_key="k", max_retries=retries, **kw)
    j._session = FakeSession(outcomes)  # type: ignore[assignment]
    return j, j._session  # type: ignore[return-value]


class FakeJev:
    """Says yes iff the generated answer contains the word 'good'. Fails on 'boom'; crashes on 'weird'."""
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, category: int, question: str, gold: str, generated: str) -> JevVerdict:
        self.calls += 1
        if "boom" in generated:
            raise JevError("HTTP 500")
        if "weird" in generated:
            raise TypeError("not a JevError")
        p = 0.9 if "good" in generated else 0.1
        return JevVerdict(correct=p > 0.5, p_yes=p, input_tokens=10, output_tokens=1)


def judged(items: list[dict], judge=None, threshold: float = 0.5, **kw) -> list[dict]:
    return apply_threshold(asyncio.run(judge_all(items, judge or FakeJev(), **kw)), threshold)


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

    def test_summary_counts_agreement_confusion_and_errors(self) -> None:
        s = summarize(judged(load_items(self.dir)))
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


class ParseResponseTest(unittest.TestCase):
    def body(self, noul, usage=None) -> dict:
        b: dict = {"answers": {QUESTION_KEY: {"noul": noul}}}
        if usage is not None:
            b["usage"] = usage
        return b

    def test_malformed_usage_never_raises_a_non_jev_error(self) -> None:
        for usage in ({"input_tokens": None}, {"input_tokens": "5"}, {"input_tokens": -1}, "oops", [1], 7):
            v = parse_response(self.body(0.9, usage))
            self.assertEqual((v.input_tokens, v.output_tokens), (0, 0), usage)
        self.assertEqual(parse_response(self.body(0.9, {"input_tokens": 4, "output_tokens": 2})).input_tokens, 4)

    def test_non_dict_body_is_a_jev_error(self) -> None:
        for bad in (None, [], "x", 3):
            with self.assertRaises(JevError):
                parse_response(bad)

    def test_nan_infinite_and_out_of_range_probabilities_are_rejected(self) -> None:
        for p in (math.nan, math.inf, -math.inf, 7, -0.1, 1.0001):
            with self.assertRaises(JevError, msg=str(p)):
                parse_response(self.body(p))
        for p in (0, 0.0, 1, 1.0, 0.5):
            self.assertEqual(parse_response(self.body(p)).p_yes, float(p))


class JevJudgeRetryTest(unittest.TestCase):
    def run_judge(self, outcomes: list, retries: int = 3, **kw):
        j, sess = _judge_with(outcomes, retries, **kw)
        sleep = mock.AsyncMock()
        with mock.patch.object(jev_judge.asyncio, "sleep", new=sleep):
            try:
                return asyncio.run(j.judge(1, "q", "g", "a")), sess, sleep
            except JevError as exc:
                exc.sess, exc.sleep = sess, sleep  # type: ignore[attr-defined]
                raise

    def test_network_error_is_retried_then_succeeds(self) -> None:
        v, sess, _ = self.run_judge([aiohttp.ClientConnectionError("reset"), asyncio.TimeoutError(), (200, OK_BODY)])
        self.assertTrue(v.correct)
        self.assertEqual(sess.calls, 3)

    def test_persistent_network_error_becomes_jev_error_after_max_retries_calls(self) -> None:
        with self.assertRaises(JevError) as cm:
            self.run_judge([asyncio.TimeoutError()] * 3)
        self.assertIn("gave up after 3", str(cm.exception))
        self.assertEqual(cm.exception.sess.calls, 3)  # type: ignore[attr-defined]
        self.assertEqual(cm.exception.sleep.await_count, 2)  # type: ignore[attr-defined]  # no trailing sleep

    def test_non_json_200_is_retried_and_then_jev_error(self) -> None:
        v, sess, _ = self.run_judge([(200, "<html>"), (200, OK_BODY)])
        self.assertTrue(v.correct)
        with self.assertRaises(JevError):
            self.run_judge([(200, "<html>")] * 3)

    def test_non_utf8_body_is_a_retried_transport_failure_not_a_crash(self) -> None:
        bad = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        v, sess, _ = self.run_judge([(200, bad), (200, OK_BODY)])
        self.assertTrue(v.correct)
        self.assertEqual(sess.calls, 2)

    def test_malformed_but_valid_json_200_is_a_permanent_jev_error_not_a_type_error(self) -> None:
        body = json.dumps({"answers": {QUESTION_KEY: {"noul": 0.9}}, "usage": {"input_tokens": None}})
        v, _, _ = self.run_judge([(200, body)])
        self.assertEqual(v.input_tokens, 0)
        with self.assertRaises(JevError) as cm:
            self.run_judge([(200, json.dumps({"answers": {QUESTION_KEY: {"noul": 7}}}))])
        self.assertEqual(cm.exception.sess.calls, 1)  # type: ignore[attr-defined]  # not retried

    def test_retryable_statuses_retry_and_other_4xx_does_not(self) -> None:
        for status in (408, 425, 429, 529, 500, 503):
            v, sess, _ = self.run_judge([(status, "x"), (200, OK_BODY)])
            self.assertEqual(sess.calls, 2, status)
        for status in (400, 401, 403, 404, 422):
            with self.assertRaises(JevError) as cm:
                self.run_judge([(status, "no"), (200, OK_BODY)])
            self.assertEqual(cm.exception.sess.calls, 1, status)  # type: ignore[attr-defined]

    def test_retry_after_header_is_honoured_and_capped(self) -> None:
        _, _, sleep = self.run_judge([(429, "slow", {"Retry-After": "7"}), (200, OK_BODY)])
        sleep.assert_awaited_once_with(7.0)
        _, _, sleep = self.run_judge([(429, "slow", {"Retry-After": "9999"}), (200, OK_BODY)])
        sleep.assert_awaited_once_with(jev_judge.MAX_BACKOFF_S)

    def test_backoff_has_jitter_and_grows(self) -> None:
        rng = random.Random(1)
        first = {backoff_delay(0, None, rng) for _ in range(20)}
        self.assertGreater(len(first), 1)                       # not deterministic: callers do not retry in lockstep
        self.assertTrue(all(0 <= d <= 1 for d in first))
        self.assertTrue(all(0 <= backoff_delay(4, None, rng) <= 16 for _ in range(20)))
        self.assertEqual(backoff_delay(0, "not-a-number", random.Random(1)), backoff_delay(0, None, random.Random(1)))

    def test_circuit_opens_after_consecutive_failures_and_closes_on_success(self) -> None:
        j, sess = _judge_with([(500, "x")] * 2 + [(200, OK_BODY)] + [(500, "x")] * 2, retries=1, breaker=2)
        async def go() -> list:
            out = []
            for _ in range(5):
                try:
                    out.append((await j.judge(1, "q", "g", "a")).correct)
                except JevCircuitOpen:
                    out.append("open")
                except JevError:
                    out.append("fail")
            return out
        with mock.patch.object(jev_judge.asyncio, "sleep", new=mock.AsyncMock()):
            self.assertEqual(asyncio.run(go()), ["fail", "fail", "open", "open", "open"])
        self.assertEqual(sess.calls, 2)  # once open, the API is not called again

        j2, sess2 = _judge_with([(500, "x"), (200, OK_BODY), (500, "x"), (200, OK_BODY)], retries=1, breaker=2)
        async def alternate() -> list:
            out = []
            for _ in range(4):
                try:
                    out.append((await j2.judge(1, "q", "g", "a")).correct)
                except JevError:
                    out.append("fail")
            return out
        self.assertEqual(asyncio.run(alternate()), ["fail", True, "fail", True])  # a success resets the count

    def test_criteria_keep_the_date_rules_from_the_original_prompt(self) -> None:
        for phrase in ("14 days", "50%", "vague reference", "'last year'"):
            self.assertIn(phrase, JEV_CRITERIA_TRUE)


class JudgeAllTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "predicted_x"
        self.dir.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_unexpected_exception_in_one_item_does_not_lose_the_others(self) -> None:
        for i, g in enumerate(["good", "weird", "good", "bad", "boom"]):
            _write(self.dir, f"conv0_q{i}", 1, "a", {"top_20": ("CORRECT", g)})
        seen: list[dict] = []
        rows = asyncio.run(judge_all(load_items(self.dir), FakeJev(), seen.append))
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(seen), 5)
        errors = {r["question_id"]: r["error"] for r in rows if "error" in r}
        self.assertEqual(set(errors), {"conv0_q1", "conv0_q4"})
        self.assertTrue(errors["conv0_q1"].startswith("TypeError"))
        self.assertEqual(sum(1 for r in rows if "jev_p_yes" in r), 3)

    def test_on_row_failure_still_awaits_every_task_before_raising(self) -> None:
        for i in range(4):
            _write(self.dir, f"conv0_q{i}", 1, "a", {"top_20": ("CORRECT", "good")})
        finished: list[str] = []

        class Slow:
            n = 0

            async def __call__(self, *a: object) -> JevVerdict:
                Slow.n += 1
                await asyncio.sleep(0 if Slow.n == 1 else 0.05)  # first row fails while the rest are in flight
                finished.append("x")
                return JevVerdict(True, 0.9, 0, 0)

        def bad_on_row(row: dict) -> None:
            raise OSError("disk full")

        with self.assertRaises(OSError):
            asyncio.run(judge_all(load_items(self.dir), Slow(), bad_on_row))
        self.assertEqual(len(finished), 4)  # nothing left running in the background

    def test_original_judge_failure_is_skipped_not_compared(self) -> None:
        _write(self.dir, "conv0_q0", 1, "a", {"top_20": ("WRONG", "good")}, reason="")  # old file: inferred
        _write(self.dir, "conv0_q1", 1, "a", {"top_20": ("CORRECT", "good")})
        s = summarize(judged(load_items(self.dir)))
        self.assertEqual((s["judged"], s["skipped_original_failed"], s["skipped_inferred"], s["errors"]), (1, 1, 1, 0))
        self.assertEqual(s["agreement"], 1.0)

    def test_judge_failed_flag_beats_the_reason_heuristic(self) -> None:
        # Terse but genuine verdict: empty reason, flag says it did not fail -> compared.
        _write(self.dir, "conv0_q0", 1, "a", {"top_20": ("CORRECT", "good")}, reason="", judge_failed=False)
        # Bad label with a non-empty reason: flag says it failed -> excluded, heuristic alone would miss it.
        _write(self.dir, "conv0_q1", 1, "a", {"top_20": ("WRONG", "good")}, reason="because", judge_failed=True)
        items = {i["question_id"]: i for i in load_items(self.dir)}
        self.assertFalse(items["conv0_q0"]["original_judge_failed"])
        self.assertTrue(items["conv0_q1"]["original_judge_failed"])
        self.assertFalse(items["conv0_q1"]["original_failure_inferred"])
        s = summarize(judged(list(items.values())))
        self.assertEqual((s["judged"], s["skipped_original_failed"], s["skipped_inferred"]), (1, 1, 0))

    def test_on_row_sees_every_new_row_as_it_finishes(self) -> None:
        _write(self.dir, "conv0_q0", 1, "a", {"top_20": ("CORRECT", "good"), "top_100": ("WRONG", "boom")})
        seen: list[dict] = []
        asyncio.run(judge_all(load_items(self.dir), FakeJev(), seen.append))
        self.assertEqual(len(seen), 2)


class SamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_limit_spreads_over_conversations_not_just_the_first_files(self) -> None:
        for conv in range(10):
            for q in range(20):
                _write(self.dir, f"conv{conv}_q{q}", 1, "a", {"top_20": ("CORRECT", "good")})
        got = select_items(load_items(self.dir), 20, seed=0)
        convs = {i["question_id"].split("_")[0] for i in got}
        self.assertEqual(len(got), 20)
        self.assertGreater(len(convs), 4)  # the old behaviour gave exactly {'conv0'}

    def test_limit_spreads_over_categories_and_cutoffs_and_is_deterministic(self) -> None:
        for i in range(6):
            _write(self.dir, f"conv0_q{i}", 1, "a", {"top_20": ("CORRECT", "good")})
        _write(self.dir, "conv1_q0", 2, "a", {"top_20": ("CORRECT", "good"), "top_100": ("CORRECT", "good")})
        got = select_items(load_items(self.dir), 3)
        self.assertEqual({(i["category"], i["cutoff"]) for i in got}, {(1, "top_20"), (2, "top_20"), (2, "top_100")})
        self.assertEqual(select_items(load_items(self.dir), 3), got)
        for seed in range(3):  # any seed still gives a full, balanced sample
            self.assertEqual(len(select_items(load_items(self.dir), 3, seed=seed)), 3)

    def test_limit_is_applied_after_excluding_failed_originals(self) -> None:
        for q in range(4):
            _write(self.dir, f"conv0_q{q}", 1, "a", {"top_20": ("WRONG", "good")}, reason="")      # failed
        for q in range(4, 8):
            _write(self.dir, f"conv0_q{q}", 1, "a", {"top_20": ("CORRECT", "good")})
        got = select_items(load_items(self.dir), 3)
        sendable = [i for i in got if not i["original_judge_failed"]]
        self.assertEqual(len(sendable), 3)  # a full N, not N minus the failures
        self.assertEqual(len(got) - len(sendable), 4)  # failures still carried so they are counted

    def test_no_limit_returns_everything(self) -> None:
        for q in range(5):
            _write(self.dir, f"conv0_q{q}", 1, "a", {"top_20": ("CORRECT", "good")})
        self.assertEqual(len(select_items(load_items(self.dir), None)), 5)

    def test_cli_rejects_zero_and_negative_limits(self) -> None:
        for bad in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                compare_judges._positive_int(bad)
        self.assertEqual(compare_judges._positive_int("5"), 5)


class ResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pred = self.root / "predicted_x"
        self.pred.mkdir()
        _write(self.pred, "conv0_q0", 1, "a", {"top_20": ("CORRECT", "good")})
        _write(self.pred, "conv0_q1", 1, "a", {"top_20": ("WRONG", "bad")})
        _write(self.pred, "conv0_q2", 1, "a", {"top_20": ("CORRECT", "boom")})

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def args(self, **kw) -> argparse.Namespace:
        base = dict(predicted_dir=str(self.pred), out=None, threshold=0.5, limit=None, seed=0,
                    concurrency=2, breaker=10, dry_run=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def run_with(self, judge: FakeJev, **kw) -> int:
        class Ctx:
            def __init__(self, **k: object) -> None: ...
            async def __aenter__(self) -> "Ctx": return self
            async def __aexit__(self, *e: object) -> None: ...
        Ctx.judge = judge  # type: ignore[attr-defined]
        with mock.patch.object(compare_judges, "JevJudge", Ctx):
            return asyncio.run(compare_judges.run(self.args(**kw)))

    def test_rerun_reuses_finished_verdicts_and_retries_only_errors(self) -> None:
        j1 = FakeJev()
        self.assertEqual(self.run_with(j1), 0)
        self.assertEqual(j1.calls, 3)
        out = self.root / "judge_compare"
        rows_file = next(out.glob("jev_rows_*.jsonl"))
        self.assertEqual(len(rows_file.read_text().splitlines()), 3)

        j2 = FakeJev()
        self.assertEqual(self.run_with(j2), 0)
        self.assertEqual(j2.calls, 1)  # only the errored 'boom' row is called again
        self.assertEqual(len(list(out.glob("jev_rows_*.jsonl"))), 1)  # same stable progress file

    def test_changing_the_threshold_needs_no_new_calls(self) -> None:
        self.run_with(FakeJev())
        j = FakeJev()
        self.run_with(j, threshold=0.95)
        self.assertEqual(j.calls, 1)  # still just the errored row
        report = json.loads(next((self.root / "judge_compare").glob("judge_compare_*.json")).read_text())
        self.assertFalse(any(r.get("jev_correct") for r in report["rows"] if "jev_p_yes" in r))  # 0.9 < 0.95

    def test_changed_input_is_rejudged(self) -> None:
        self.run_with(FakeJev())
        _write(self.pred, "conv0_q0", 1, "a", {"top_20": ("CORRECT", "good, edited")})
        j = FakeJev()
        self.run_with(j)
        self.assertEqual(j.calls, 2)  # the edited answer and the errored one

    def test_run_id_is_stable_and_depends_on_source(self) -> None:
        self.assertEqual(run_id(self.pred), run_id(self.pred))
        other = self.root / "predicted_y"
        other.mkdir()
        self.assertNotEqual(run_id(self.pred), run_id(other))

    def test_load_progress_ignores_garbage_and_error_rows(self) -> None:
        f = self.root / "p.jsonl"
        f.write_text("\n".join([
            json.dumps({"question_id": "a", "cutoff": "t", "jev_p_yes": 0.7}),
            "not json at all",
            json.dumps({"question_id": "b", "cutoff": "t", "error": "boom"}),
            json.dumps({"question_id": "c"}),
        ]))
        self.assertEqual(set(load_progress(f)), {("a", "t")})

    def test_run_writes_report_outside_the_result_dir_and_dry_run_makes_no_calls(self) -> None:
        j = FakeJev()
        self.assertEqual(self.run_with(j, dry_run=True), 0)
        self.assertEqual(j.calls, 0)
        self.assertFalse((self.root / "judge_compare").exists())
        self.run_with(FakeJev())
        self.assertEqual(list(self.pred.glob("judge_compare*")), [])  # result dir untouched
        out = self.root / "judge_compare"
        self.assertEqual(len(list(out.glob("*.md"))), 1)
        self.assertIn("p>0.3", next(out.glob("*.md")).read_text())

    def test_dry_run_reports_skipped_and_prints_a_sendable_first_request(self) -> None:
        _write(self.pred, "conv0_q0", 1, "a", {"top_20": ("WRONG", "good")}, reason="")  # failed original, sorts first
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(self.run_with(FakeJev(), dry_run=True), 0)
        text = buf.getvalue()
        self.assertIn("2 answers to judge, 1 skipped", text)
        self.assertNotIn('"generated_answer": "good"', text.split("First request:")[1].split("\n")[0])

    def test_missing_api_key_is_a_clear_exit_code_2(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(asyncio.run(compare_judges.run(self.args())), 2)


if __name__ == "__main__":
    unittest.main()
