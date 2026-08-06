import json

from conductor import browsecomp, db, metrics, reuse


def test_ready_browsecomp_store_can_be_adopted_and_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDUCTOR_DB", str(tmp_path / "evals.db"))
    monkeypatch.setenv("CONDUCTOR_DATA_ROOT", str(tmp_path / "data"))
    db._initialized = False
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    gold = tmp_path / "gold.jsonl"
    for path in (corpus, queries, qrels, gold):
        path.write_text("", encoding="utf-8")
    source_manifest = tmp_path / "store.json"
    source_manifest.write_text(
        json.dumps(
            {
                "benchmark": "browsecomp_plus",
                "status": "READY",
                "canonical": False,
                "corpus": {"revision": "pilot-v1", "sha256": "abc", "document_count": 1000},
                "scope": {"user_id": "pilot-user"},
                "documents": {"id_namespace": "0d524176-aac3-5b3d-80c5-07029a84a668"},
            }
        ),
        encoding="utf-8",
    )

    adopted = browsecomp.adopt_store(
        manifest_path=str(source_manifest),
        host="https://staging.example",
        corpus_path=str(corpus),
        queries_path=str(queries),
        qrels_path=str(qrels),
        gold_qrels_path=str(gold),
        artifact_id="bcp-pilot",
    )
    loaded = reuse.load_store("bcp-pilot")

    assert adopted["artifact_id"] == "bcp-pilot"
    assert loaded["manifest"]["scope"]["user_id"] == "pilot-user"
    assert loaded["manifest"]["host"] == "https://staging.example"


def test_browsecomp_metrics_keep_quality_latency_and_failure_count(tmp_path):
    result = tmp_path / "results.json"
    result.write_text(
        json.dumps(
            {
                "search_mode": "fast",
                "query_count": 20,
                "failure_count": 1,
                "canonical": False,
                "metrics": {"evidence": {"recall_at_10": 0.7, "ndcg_at_10": 0.6}},
                "latency_ms": {"p50": 40.0, "p95": 75.0, "mean": 44.0},
            }
        ),
        encoding="utf-8",
    )

    extracted = metrics.extract_browsecomp_plus(str(result))

    assert extracted["pairs"]["ndcg_at_10"] == 0.6
    assert extracted["pairs"]["latency_p95_ms"] == 75.0
    assert extracted["failure_count"] == 1
    assert extracted["canonical"] is False
