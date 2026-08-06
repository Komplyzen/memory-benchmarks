from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_pilot(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    gold_qrels_path: Path,
    output_dir: Path,
    query_count: int = 20,
    document_count: int = 1000,
    seed: int = 7,
) -> dict:
    queries = {str(row["id"]): row for row in read_jsonl(queries_path)}
    qrels = list(read_jsonl(qrels_path))
    gold_qrels = list(read_jsonl(gold_qrels_path))
    eligible = sorted(
        set(queries)
        & {str(row["query_id"]) for row in qrels}
        & {str(row["query_id"]) for row in gold_qrels}
    )
    if query_count > len(eligible):
        raise ValueError(f"Requested {query_count} queries but only {len(eligible)} are eligible")
    rng = random.Random(seed)
    selected_queries = sorted(rng.sample(eligible, query_count))
    selected_set = set(selected_queries)
    selected_qrels = [row for row in qrels if str(row["query_id"]) in selected_set]
    selected_gold = [row for row in gold_qrels if str(row["query_id"]) in selected_set]
    evidence_ids = {
        str(row["doc_id"])
        for row in [*selected_qrels, *selected_gold]
        if int(row.get("relevance", 0)) > 0
    }
    if len(evidence_ids) > document_count:
        raise ValueError(f"Selected queries require {len(evidence_ids)} evidence documents, above pilot size {document_count}")

    evidence_rows: dict[str, dict] = {}
    distractors: list[dict] = []
    distractor_target = document_count - len(evidence_ids)
    seen_distractors = 0
    for row in read_jsonl(corpus_path):
        docid = str(row.get("id") or row.get("docid") or "")
        normalized = {"id": docid, "text": row["text"]}
        if docid in evidence_ids:
            evidence_rows[docid] = normalized
            continue
        seen_distractors += 1
        if len(distractors) < distractor_target:
            distractors.append(normalized)
        else:
            replacement = rng.randrange(seen_distractors)
            if replacement < distractor_target:
                distractors[replacement] = normalized

    missing = sorted(evidence_ids - set(evidence_rows))
    if missing:
        raise ValueError(f"Evidence documents are missing from the corpus: {missing[:10]}")
    documents = sorted([*evidence_rows.values(), *distractors], key=lambda row: row["id"])
    if len(documents) != document_count or len({row["id"] for row in documents}) != document_count:
        raise ValueError("Pilot did not produce the requested number of unique documents")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "corpus.jsonl", documents)
    write_jsonl(output_dir / "queries.jsonl", [queries[query_id] for query_id in selected_queries])
    write_jsonl(output_dir / "qrels.jsonl", selected_qrels)
    write_jsonl(output_dir / "gold_qrels.jsonl", selected_gold)
    manifest = {
        "benchmark": "browsecomp_plus",
        "canonical": False,
        "note": "Reduced-corpus pilot. It validates wiring and estimates cost; its scores are not official full-corpus results.",
        "seed": seed,
        "query_count": len(selected_queries),
        "document_count": len(documents),
        "evidence_document_count": len(evidence_ids),
        "query_ids": selected_queries,
        "corpus_sha256": hashlib.sha256((output_dir / "corpus.jsonl").read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic BrowseComp-Plus pilot corpus")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--gold-qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--document-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = build_pilot(
        corpus_path=args.corpus,
        queries_path=args.queries,
        qrels_path=args.qrels,
        gold_qrels_path=args.gold_qrels,
        output_dir=args.output_dir,
        query_count=args.query_count,
        document_count=args.document_count,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
