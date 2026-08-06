from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .dataset import read_jsonl
from .platform_client import PlatformClient


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    by_query: dict[str, dict[str, int]] = {}
    for row in read_jsonl(path):
        by_query.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = int(row["relevance"])
    return by_query


def _recall(retrieved: list[str], relevant: dict[str, int]) -> float:
    positives = {docid for docid, relevance in relevant.items() if relevance > 0}
    return len(set(retrieved) & positives) / len(positives) if positives else 0.0


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _ndcg(ranked: list[str], relevant: dict[str, int], cutoff: int) -> float:
    unique_ranked = list(dict.fromkeys(ranked))
    gains = [relevant.get(docid, 0) for docid in unique_ranked[:cutoff]]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(relevant.values(), reverse=True)[:cutoff]
    idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _parse_server_timing(header: str | None) -> dict[str, float]:
    timings: dict[str, float] = {}
    for part in (header or "").split(","):
        fields = [field.strip() for field in part.split(";")]
        if not fields or not fields[0]:
            continue
        duration = next((field[4:] for field in fields[1:] if field.startswith("dur=")), None)
        if duration is None:
            continue
        try:
            timings[fields[0]] = float(duration)
        except ValueError:
            continue
    return timings


def _timing_summary(records: list[dict]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for record in records:
        for name, duration in _parse_server_timing(record.get("server_timing")).items():
            values.setdefault(name, []).append(duration)
    return {
        name: {
            "mean": round(statistics.mean(durations), 3),
            "p50": _percentile(durations, 0.5),
            "p95": _percentile(durations, 0.95),
            "sample_count": len(durations),
        }
        for name, durations in sorted(values.items())
    }


def _usage_cost(record: dict) -> float:
    usage = record.get("usage") or {}
    return float(usage.get("estimated_cost_usd") or 0.0)


def _require_cost_budget(*, spent: float, reserve: float, cap: float) -> None:
    if spent + reserve > cap:
        raise RuntimeError(
            f"agentic cost gate stopped before the next query: spent=${spent:.4f}, "
            f"reserve=${reserve:.4f}, cap=${cap:.4f}"
        )


def _validate_inputs(queries: list[dict], corpus: Path, qrel_sets: dict[str, dict[str, dict[str, int]]]) -> None:
    query_ids = {str(row["id"]) for row in queries}
    corpus_ids = {str(row.get("id") or row.get("docid")) for row in read_jsonl(corpus)}
    for label, qrels in qrel_sets.items():
        extra_queries = set(qrels) - query_ids
        if extra_queries:
            raise RuntimeError(f"{label} contains {len(extra_queries)} queries outside this run")
        required_docs = {docid for by_doc in qrels.values() for docid, relevance in by_doc.items() if relevance > 0}
        missing_docs = required_docs - corpus_ids
        if missing_docs:
            raise RuntimeError(f"{label} references {len(missing_docs)} documents missing from the corpus")


def _chunk_count(text: str, *, size: int, overlap: int) -> int:
    count = 0
    start = 0
    while start < len(text):
        target = min(start + size, len(text))
        end = target
        if target < len(text):
            boundary = text.rfind(" ", start + (size // 2), target)
            if boundary > start:
                end = boundary
        if text[start:end].strip():
            count += 1
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return count


def _memory_map(corpus: Path, store_manifest: dict) -> dict[str, str]:
    namespace = uuid.UUID(store_manifest["documents"]["id_namespace"])
    revision = store_manifest["corpus"]["revision"]
    chunking = store_manifest["documents"].get("chunking")
    mapped: dict[str, str] = {}
    for row in read_jsonl(corpus):
        docid = str(row.get("id") or row.get("docid"))
        if chunking:
            count = _chunk_count(row["text"], size=int(chunking["size"]), overlap=int(chunking["overlap"]))
            for chunk_index in range(count):
                mapped[str(uuid.uuid5(namespace, f"{revision}:{docid}:{chunk_index}"))] = docid
        else:
            mapped[str(uuid.uuid5(namespace, f"{revision}:{docid}"))] = docid
    return mapped


def _search_record(
    *, client: PlatformClient, mode: str, row: dict, top_k: int, memory_to_docid: dict[str, str]
) -> dict:
    query_id = str(row["id"])
    started = time.perf_counter()
    try:
        response = client.search(mode=mode, query=row["text"], top_k=top_k)
        results = response.body.get("results") or []
        ranked_docids = []
        unmapped_ids = []
        for result in results:
            memory_id = str(result.get("id") or "")
            metadata = result.get("metadata") or {}
            docid = metadata.get("docid") or memory_to_docid.get(memory_id)
            if docid is not None:
                ranked_docids.append(str(docid))
            elif memory_id:
                unmapped_ids.append(memory_id)
        retrieved_ids = response.body.get("retrieved_memory_ids") or [result.get("id") for result in results]
        unmapped_ids.extend(str(memory_id) for memory_id in retrieved_ids if str(memory_id) not in memory_to_docid)
        if unmapped_ids:
            raise RuntimeError(f"response contained {len(set(unmapped_ids))} memory IDs without source-document mappings")
        retrieved_docids = [memory_to_docid[str(memory_id)] for memory_id in retrieved_ids]
        return {
            "query_id": query_id,
            "query": row["text"],
            "status": "ok",
            "wall_ms": response.wall_ms,
            "server_timing": response.server_timing,
            "ranked_docids": ranked_docids,
            "retrieved_docids": retrieved_docids,
            "answer": response.body.get("answer"),
            "usage": response.body.get("usage"),
        }
    except Exception as exc:
        return {
            "query_id": query_id,
            "query": row["text"],
            "status": "failed",
            "wall_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "ranked_docids": [],
            "retrieved_docids": [],
        }


def _build_result(
    *,
    mode: str,
    top_k: int,
    records: list[dict],
    evidence: dict[str, dict[str, int]],
    gold: dict[str, dict[str, int]],
    store_manifest: dict,
    store_manifest_path: Path,
    corpus_revision: str,
    user_id: str,
    platform_sha: str | None,
    agentic_contract_sha256: str | None,
) -> dict:
    successful = [record for record in records if record["status"] == "ok"]

    def metric_block(qrels: dict[str, dict[str, int]]) -> dict:
        metrics = {
            f"recall_at_{cutoff}": round(
                statistics.mean(
                    _recall(_dedupe(record["ranked_docids"])[:cutoff], qrels.get(record["query_id"], {}))
                    for record in records
                ),
                6,
            ) if records else None
            for cutoff in (5, 10)
        }
        metrics["ndcg_at_10"] = round(
            statistics.mean(_ndcg(record["ranked_docids"], qrels.get(record["query_id"], {}), 10) for record in records),
            6,
        ) if records else None
        if mode == "agentic":
            metrics["recall_over_all_retrieved"] = round(
                statistics.mean(
                    _recall(_dedupe(record["retrieved_docids"]), qrels.get(record["query_id"], {}))
                    for record in records
                ),
                6,
            ) if records else None
        return metrics

    latencies = [record["wall_ms"] for record in successful]
    return {
        "benchmark": "browsecomp_plus",
        "search_mode": mode,
        "canonical": bool(store_manifest.get("canonical", False)),
        "store_manifest": str(store_manifest_path),
        "store_manifest_sha256": hashlib.sha256(store_manifest_path.read_bytes()).hexdigest(),
        "platform_sha": platform_sha,
        "agentic_contract_sha256": agentic_contract_sha256,
        "corpus_revision": corpus_revision,
        "scope": {"user_id": user_id},
        "top_k": top_k,
        "query_count": len(records),
        "success_count": len(successful),
        "failure_count": len(records) - len(successful),
        "metrics": {"evidence": metric_block(evidence), "gold": metric_block(gold)},
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "server_timing_ms": _timing_summary(successful),
        "usage": {
            "estimated_cost_usd": round(sum(_usage_cost(record) for record in successful), 6),
            "prompt_tokens": sum(int((record.get("usage") or {}).get("prompt_tokens") or 0) for record in successful),
            "completion_tokens": sum(int((record.get("usage") or {}).get("completion_tokens") or 0) for record in successful),
            "iterations": sum(int((record.get("usage") or {}).get("iterations") or 0) for record in successful),
            "tool_calls": sum(int((record.get("usage") or {}).get("tool_calls") or 0) for record in successful),
            "search_calls": sum(int((record.get("usage") or {}).get("search_calls") or 0) for record in successful),
            "models": sorted({str((record.get("usage") or {}).get("model")) for record in successful if (record.get("usage") or {}).get("model")}),
        },
        "records": records,
    }


def run(args) -> Path:
    store_manifest = json.loads(args.store_manifest.read_text())
    if store_manifest.get("status") != "READY":
        raise RuntimeError("Store manifest is not READY")
    if store_manifest["corpus"]["revision"] != args.corpus_revision:
        raise RuntimeError("Corpus revision differs from the reusable store")
    queries = list(read_jsonl(args.queries))
    if args.max_queries:
        queries = queries[: args.max_queries]
    evidence = _load_qrels(args.qrels)
    gold = _load_qrels(args.gold_qrels)
    _validate_inputs(queries, args.corpus, {"evidence qrels": evidence, "gold qrels": gold})
    memory_to_docid = _memory_map(args.corpus, store_manifest)
    api_key = args.api_key or os.getenv("MEM0_API_KEY")
    if not api_key:
        raise RuntimeError("MEM0_API_KEY is required")
    user_id = store_manifest["scope"]["user_id"]
    client = PlatformClient(args.mem0_host, api_key, user_id, timeout=args.timeout)

    output_dir = args.output_dir or Path("results/browsecomp_plus") / f"predicted_{args.project_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    completed: dict[str, dict] = {}
    if progress_path.exists():
        for row in read_jsonl(progress_path):
            completed[str(row["query_id"])] = row

    def search(row: dict) -> dict:
        return _search_record(
            client=client,
            mode=args.search_mode,
            row=row,
            top_k=args.top_k,
            memory_to_docid=memory_to_docid,
        )

    def save_record(record: dict) -> None:
        completed[record["query_id"]] = record
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"queries {len(completed)}/{len(queries)}", flush=True)

    pending = [row for row in queries if str(row["id"]) not in completed]
    if args.max_cost_usd is not None:
        spent = sum(_usage_cost(record) for record in completed.values())
        for row in pending:
            _require_cost_budget(spent=spent, reserve=args.cost_reserve_usd, cap=args.max_cost_usd)
            record = search(row)
            save_record(record)
            spent += _usage_cost(record)
            if spent > args.max_cost_usd:
                raise RuntimeError(f"agentic cost cap exceeded after a response: spent=${spent:.4f}")
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(search, row): str(row["id"]) for row in pending}
            for future in as_completed(futures):
                record = future.result()
                save_record(record)

    records = [completed[str(row["id"])] for row in queries]
    result = _build_result(
        mode=args.search_mode,
        top_k=args.top_k,
        records=records,
        evidence=evidence,
        gold=gold,
        store_manifest=store_manifest,
        store_manifest_path=args.store_manifest,
        corpus_revision=args.corpus_revision,
        user_id=user_id,
        platform_sha=args.platform_sha,
        agentic_contract_sha256=args.agentic_contract_sha256,
    )
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to: {result_path}")
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BrowseComp-Plus against one platform search mode")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--search-mode", choices=["regular", "fast", "agentic"], required=True)
    parser.add_argument("--store-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--gold-qrels", type=Path, required=True)
    parser.add_argument("--mem0-host", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--cost-reserve-usd", type=float, default=1.25)
    parser.add_argument("--platform-sha")
    parser.add_argument("--agentic-contract-sha256")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    limits = {"regular": 1000, "fast": 100, "agentic": 50}
    if not 1 <= args.top_k <= limits[args.search_mode]:
        raise SystemExit(f"--top-k for {args.search_mode} must be between 1 and {limits[args.search_mode]}")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    if args.max_cost_usd is not None and (args.search_mode != "agentic" or args.concurrency != 1):
        raise SystemExit("--max-cost-usd requires agentic mode with concurrency 1")
    if args.max_cost_usd is not None and (args.max_cost_usd <= 0 or args.cost_reserve_usd <= 0):
        raise SystemExit("cost cap and reserve must be positive")
    run(args)


if __name__ == "__main__":
    main()
