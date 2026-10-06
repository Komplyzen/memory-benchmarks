#!/usr/bin/env python3
"""
KG smoke test
=============

Against a running Komplyt KG: ingest one fake dataset session of three turns,
run one read-only search twice (ids must match), print what came back, then
wipe the project.

    KG_DB_NAME=bench python -m scripts.kg_smoke [--kg-url http://localhost:3000] [--no-wipe]
    KG_DB_NAME=bench python -m scripts.kg_smoke --stub-agent     # no LLM: scripted tool calls (CI)

Default ingest is the MCP agent (needs OPENAI_API_KEY or ANTHROPIC_API_KEY for
--agent-provider). --stub-agent replaces it with a fixed sequence: session
start, one kg_query, three kg_node_create calls (one relates_to), one kg_node_update,
session end, all through the same guardrails.

Needs: a KG HTTP server, a token (`kg auth login` or KG_TOKEN), and root
SurrealDB env (KG_DB_URL/KG_DB_USER/KG_DB_PASS/KG_DB_NS/KG_DB_NAME) for the wipe.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid

from benchmarks.common.kg_agent_ingest import ScriptedIngestor
from benchmarks.common.kg_client import KgClient
from benchmarks.common.mem0_client import format_search_results

TURNS = [
    {"role": "user", "content": "Caroline: I finally booked the trip to Lisbon for the second week of October. Three nights, staying near Alfama."},
    {"role": "assistant", "content": "Melanie: Nice! Are you going alone or with Tom?"},
    {"role": "user", "content": "Caroline: With Tom. It's his first time in Portugal, and we decided to skip Porto this time and just do Lisbon properly."},
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-url", default=None)
    parser.add_argument("--no-wipe", action="store_true")
    parser.add_argument("--no-spread", action="store_true")
    parser.add_argument("--stub-agent", action="store_true", help="scripted tool calls, no LLM")
    parser.add_argument("--agent-model", default="gpt-5-mini")
    parser.add_argument("--agent-provider", default="openai", choices=["openai", "anthropic", "azure", "mistral"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    user_id = f"smoke_{uuid.uuid4().hex[:8]}"
    session_epoch = 1683547200  # 2023-05-08 12:00 UTC

    kg = KgClient(
        url=args.kg_url,
        spread=not args.no_spread,
        decay="on",
        agent_model=args.agent_model,
        agent_provider=args.agent_provider,
        ingestor=ScriptedIngestor() if args.stub_agent else None,
    )
    async with kg:
        print(f"user_id={user_id} project={kg.project_for(user_id)} url={kg.url} ingest={kg.ingest_mode}")

        contract = await kg.tool_contract()
        print(f"tool contract: {len(contract.raw_tools)} tools sha256={contract.sha256[:16]} instructions={len(contract.instructions)} chars")

        t0 = time.monotonic()
        added = await kg.add(TURNS, user_id, timestamp=session_epoch)
        print(f"\nADD ({(time.monotonic() - t0) * 1000:.0f} ms):")
        # Every check below is a hard failure: smoke.sh / leg.sh / the mcp-smoke PR job decide
        # pass/fail purely on this process's exit code (test-architect review, LYT-292).
        # add() buffers the session and flushes at the first search; an empty result list here is normal.
        if added is None or not isinstance(added, dict):
            fail("add() failed - check the log above (token? session? agent?)")
        for r in added.get("results") or []:
            print(f"  {r['id']}  {r['memory'][:120]}")
        print(f"  ingest stats: {json.dumps(kg.ingestor.stats.as_metadata())}")
        print(f"  guardrails:   {json.dumps(kg.guardrails)}")

        t0 = time.monotonic()
        raw = await kg.search("Where is Caroline travelling and with whom?", user_id, top_k=10)
        formatted, query_debug = format_search_results(raw)
        print(f"\nSEARCH ({(time.monotonic() - t0) * 1000:.0f} ms) read_only=true:")
        print(f"  query_debug={json.dumps(query_debug)}")
        for f in formatted:
            dbg = f.get("score_debug", {})
            print(f"  score={f['score']:.0f} src={dbg.get('activation_source')} combined={dbg.get('combined_score')} decay={dbg.get('decay_score')}  {f['memory'][:120]}")

        raw2 = await kg.search("Where is Caroline travelling and with whom?", user_id, top_k=10)
        if not isinstance(raw, dict) or not isinstance(raw2, dict):
            fail(f"search returned a non-dict response: {type(raw).__name__} / {type(raw2).__name__}")
        ids1 = [r["id"] for r in raw.get("results") or []]
        ids2 = [r["id"] for r in raw2.get("results") or []]
        if not ids1:
            fail("search returned zero results after ingest (the buffered session flushes at the first search)")
        created = int((kg.ingest_metadata() or {}).get("kg_nodes_created") or 0) if hasattr(kg, "ingest_metadata") else len(ids1)
        if created <= 0:
            fail("ingest reported zero nodes created")
        print(f"\nREPEAT SEARCH identical ids: {ids1 == ids2}")
        if ids1 != ids2:
            fail(f"read_only search is not order-independent: {ids1} != {ids2}")

        if args.no_wipe:
            print("\nSkipping wipe (--no-wipe)")
        else:
            ok = await kg.delete_user(user_id)
            print(f"\nWIPE project {kg.project_for(user_id)}: {'ok' if ok else 'FAILED'}")
            if not ok:
                fail("wipe failed")
        print(f"\nPASS: {created} nodes written, {len(ids1)} retrieved read-only twice with identical ids"
              + ("" if args.no_wipe else ", project wiped"))


def fail(msg: str) -> None:
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
