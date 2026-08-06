import json
from pathlib import Path

from benchmarks.browsecomp_plus.dataset import build_pilot, read_jsonl
import pytest

from benchmarks.browsecomp_plus.run import (
    _chunk_count,
    _dedupe,
    _ndcg,
    _parse_server_timing,
    _recall,
    _require_cost_budget,
    _validate_inputs,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pilot_contains_all_selected_evidence_and_is_deterministic(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    gold = tmp_path / "gold.jsonl"
    _write(corpus, [{"id": f"doc-{index}", "text": f"text {index}"} for index in range(20)])
    _write(queries, [{"id": f"q-{index}", "text": f"query {index}"} for index in range(4)])
    _write(
        qrels,
        [{"query_id": f"q-{index}", "doc_id": f"doc-{index}", "relevance": 1} for index in range(4)],
    )
    _write(
        gold,
        [{"query_id": f"q-{index}", "doc_id": f"doc-{index + 4}", "relevance": 1} for index in range(4)],
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_pilot(
        corpus_path=corpus,
        queries_path=queries,
        qrels_path=qrels,
        gold_qrels_path=gold,
        output_dir=first,
        query_count=2,
        document_count=10,
        seed=7,
    )
    build_pilot(
        corpus_path=corpus,
        queries_path=queries,
        qrels_path=qrels,
        gold_qrels_path=gold,
        output_dir=second,
        query_count=2,
        document_count=10,
        seed=7,
    )

    selected_docs = {row["id"] for row in read_jsonl(first / "corpus.jsonl")}
    required_docs = {
        row["doc_id"]
        for path in (first / "qrels.jsonl", first / "gold_qrels.jsonl")
        for row in read_jsonl(path)
    }
    assert manifest["canonical"] is False
    assert len(selected_docs) == 10
    assert required_docs <= selected_docs
    assert (first / "corpus.jsonl").read_bytes() == (second / "corpus.jsonl").read_bytes()


def test_retrieval_metrics_use_document_level_relevance():
    qrels = {"doc-a": 2, "doc-b": 1}

    assert _recall(["doc-b", "irrelevant", "doc-a"], qrels) == 1.0
    assert _ndcg(["doc-a", "doc-b"], qrels, 2) == 1.0
    assert _ndcg(["doc-a", "doc-a"], qrels, 2) < 1.0


def test_chunk_results_are_deduplicated_to_source_documents_before_cutoff():
    ranked = ["doc-a", "doc-a", "doc-b", "doc-c"]

    assert _dedupe(ranked)[:3] == ["doc-a", "doc-b", "doc-c"]


def test_input_validation_rejects_qrels_for_documents_outside_the_pilot(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write(corpus, [{"id": "doc-a", "text": "text"}])

    with pytest.raises(RuntimeError, match="missing from the corpus"):
        _validate_inputs(
            [{"id": "q-1", "text": "query"}],
            corpus,
            {"evidence qrels": {"q-1": {"doc-missing": 1}}},
        )


def test_server_timing_parser_keeps_named_durations():
    assert _parse_server_timing("auth;dur=1.7, embedding;dur=16.2, ignored") == {
        "auth": 1.7,
        "embedding": 16.2,
    }


def test_agentic_cost_gate_reserves_budget_for_the_next_query():
    _require_cost_budget(spent=6.5, reserve=1.25, cap=8.0)

    with pytest.raises(RuntimeError, match="cost gate stopped"):
        _require_cost_budget(spent=6.8, reserve=1.25, cap=8.0)


def test_chunk_count_matches_importer_boundaries():
    text = " ".join(f"word-{index}" for index in range(2500))

    assert _chunk_count(text, size=8000, overlap=800) > 1
    assert _chunk_count("short source", size=8000, overlap=800) == 1
