#!/usr/bin/env python3
"""
KG smoke test
=============

Against a running Komplyt KG: ingest three fake turns (one dataset session),
run one read-only search, print what came back, then wipe the project.

    KG_DB_NAME=bench python -m scripts.kg_smoke [--kg-url http://localhost:3000] [--no-wipe]

Needs: a KG HTTP server, a token (`kg auth login` or KG_TOKEN), an OpenAI key
for the extractor, and root SurrealDB env (KG_DB_URL/KG_DB_USER/KG_DB_PASS/
KG_DB_NS/KG_DB_NAME) for the wipe.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid

from benchmarks.common.kg_client import KgClient
from benchmarks.common.mem0_client import format_search_results

TURNS = [
    {"role": "user", "name": "Caroline", "content": "I finally booked the trip to Lisbon for the second week of October. Three nights, staying near Alfama."},
    {"role": "user", "name": "Melanie", "content": "Nice! Are you going alone or with Tom?"},
    {"role": "user", "name": "Caroline", "content": "With Tom. It's his first time in Portugal, and we decided to skip Porto this time and just do Lisbon properly."},
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-url", default=None)
    parser.add_argument("--no-wipe", action="store_true")
    parser.add_argument("--no-spread", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    user_id = f"smoke_{uuid.uuid4().hex[:8]}"
    session_epoch = 1683547200  # 2023-05-08 12:00 UTC

    async with KgClient(url=args.kg_url, spread=not args.no_spread, decay="on") as kg:
        print(f"user_id={user_id} project={kg.project_for(user_id)} url={kg.url}")

        t0 = time.monotonic()
        added = await kg.add(TURNS, user_id, timestamp=session_epoch)
        print(f"\nADD ({(time.monotonic() - t0) * 1000:.0f} ms):")
        if added is None:
            print("  add() returned None - check the log above (token? session? extractor?)")
            return
        for r in added["results"]:
            print(f"  {r['id']}  edges={r.get('edges_created', 0)}  {r['memory']}")

        t0 = time.monotonic()
        raw = await kg.search("Where is Caroline travelling and with whom?", user_id, top_k=10)
        formatted, query_debug = format_search_results(raw)
        print(f"\nSEARCH ({(time.monotonic() - t0) * 1000:.0f} ms) read_only=true:")
        print(f"  query_debug={json.dumps(query_debug)}")
        for f in formatted:
            dbg = f.get("score_debug", {})
            print(f"  score={f['score']:.0f} src={dbg.get('activation_source')} combined={dbg.get('combined_score')} decay={dbg.get('decay_score')}  {f['memory']}")

        # Second search must return the same ids (read_only => no activation drift).
        raw2 = await kg.search("Where is Caroline travelling and with whom?", user_id, top_k=10)
        ids1 = [r["id"] for r in raw["results"]] if isinstance(raw, dict) else []
        ids2 = [r["id"] for r in raw2["results"]] if isinstance(raw2, dict) else []
        print(f"\nREPEAT SEARCH identical ids: {ids1 == ids2}")

        if args.no_wipe:
            print("\nSkipping wipe (--no-wipe)")
            return
        ok = await kg.delete_user(user_id)
        print(f"\nWIPE project {kg.project_for(user_id)}: {'ok' if ok else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
