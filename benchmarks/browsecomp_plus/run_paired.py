from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .dataset import read_jsonl
from .platform_client import PlatformClient
from .run import _build_result, _load_qrels, _memory_map, _search_record, _validate_inputs


def run(args) -> dict[str, Path]:
    store_manifest = json.loads(args.store_manifest.read_text())
    if store_manifest.get("status") != "READY":
        raise RuntimeError("Store manifest is not READY")
    if store_manifest["corpus"]["revision"] != args.corpus_revision:
        raise RuntimeError("Corpus revision differs from the reusable store")

    queries = list(read_jsonl(args.queries))
    evidence = _load_qrels(args.qrels)
    gold = _load_qrels(args.gold_qrels)
    _validate_inputs(queries, args.corpus, {"evidence qrels": evidence, "gold qrels": gold})
    memory_to_docid = _memory_map(args.corpus, store_manifest)
    api_key = args.api_key or os.getenv("MEM0_API_KEY")
    if not api_key:
        raise RuntimeError("MEM0_API_KEY is required")
    user_id = store_manifest["scope"]["user_id"]
    client = PlatformClient(args.mem0_host, api_key, user_id, timeout=args.timeout)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    records: dict[str, list[dict]] = {"regular": [], "fast": []}
    progress = {mode: (args.output_dir / mode / "progress.jsonl") for mode in records}
    for path in progress.values():
        path.parent.mkdir(parents=True)

    for index, row in enumerate(queries):
        order = ("regular", "fast") if index % 2 == 0 else ("fast", "regular")
        for mode in order:
            record = _search_record(
                client=client,
                mode=mode,
                row=row,
                top_k=args.top_k,
                memory_to_docid=memory_to_docid,
            )
            records[mode].append(record)
            with progress[mode].open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["status"] != "ok":
                raise RuntimeError(f"{mode} failed for query {record['query_id']}: {record.get('error')}")
        print(f"paired queries {index + 1}/{len(queries)}", flush=True)

    paths = {}
    for mode in ("regular", "fast"):
        result = _build_result(
            mode=mode,
            top_k=args.top_k,
            records=records[mode],
            evidence=evidence,
            gold=gold,
            store_manifest=store_manifest,
            store_manifest_path=args.store_manifest,
            corpus_revision=args.corpus_revision,
            user_id=user_id,
            platform_sha=args.platform_sha,
            agentic_contract_sha256=None,
        )
        result["run_order"] = "alternating_by_query; even=regular-first; odd=fast-first"
        path = args.output_dir / mode / "results.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        paths[mode] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired regular/fast BrowseComp-Plus staging queries")
    parser.add_argument("--store-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--gold-qrels", type=Path, required=True)
    parser.add_argument("--mem0-host", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform-sha")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 100:
        raise SystemExit("--top-k must be between 1 and 100")
    run(args)


if __name__ == "__main__":
    main()
