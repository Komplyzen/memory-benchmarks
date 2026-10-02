"""Tests for the Jev judge comparison. Run: python -m unittest scripts.test_compare_judges

The Jev HTTP call is the only thing faked (an injected judge function); loading real result files,
verdict parsing and the statistics all run for real.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.common.jev_judge import JevError, JevVerdict, build_request, parse_response, QUESTION_KEY
from scripts.compare_judges import cohens_kappa, judge_all, load_items, summarize


def _write(dirpath: Path, qid: str, cat: int, gold: str, cutoffs: dict[str, tuple[str, str]]) -> None:
    (dirpath / f"{qid}.json").write_text(json.dumps({
        "question_id": qid, "category": cat, "category_name": "single-hop", "question": f"q {qid}?",
        "ground_truth_answer": gold,
        "cutoff_results": {k: {"judgment": j, "generated_answer": g, "reason": "r"} for k, (j, g) in cutoffs.items()},
    }))


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


if __name__ == "__main__":
    unittest.main()
