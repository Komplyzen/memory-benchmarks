#!/usr/bin/env python3
"""
Slice the LongMemEval dataset into shard files (spec-benchmark-ci CAP-5)
========================================================================

    python -m scripts.slice_dataset --benchmark longmemeval --shard-size 25 --out-dir DIR [--dataset-path FILE]

Writes ``longmemeval_s_shard<idx>.json`` (blocks of --shard-size questions, in
dataset order) that the runner accepts unchanged via ``--dataset-path``.
Prints a JSON list ``[{"idx", "path", "questions"}]`` on stdout.

For locomo the shards are conversations, so nothing is written and an empty
list is printed (exit 0).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=["locomo", "longmemeval"])
    p.add_argument("--shard-size", type=int, default=25)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-path", default=None)
    args = p.parse_args()

    if args.benchmark == "locomo":
        print("[]")
        return 0
    if args.shard_size <= 0:
        print("slice: --shard-size must be positive", file=sys.stderr)
        return 2

    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        from benchmarks.longmemeval.run import DEFAULT_DATASET_DIR, download_dataset
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
        dataset_path = download_dataset(DEFAULT_DATASET_DIR, logging.getLogger("slice"))

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("slice: dataset must be a JSON list of questions", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    out = []
    for idx in range((len(data) + args.shard_size - 1) // args.shard_size):
        block = data[idx * args.shard_size:(idx + 1) * args.shard_size]
        path = os.path.join(args.out_dir, f"longmemeval_s_shard{idx}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(block, f, ensure_ascii=False)
        out.append({"idx": idx, "path": path, "questions": len(block)})
    print(f"slice: {len(data)} questions -> {len(out)} shards of {args.shard_size} in {args.out_dir}", file=sys.stderr)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
