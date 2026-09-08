# Running the benchmarks against the Komplyt KG

This fork adds a `kg` backend so the unmodified LoCoMo and LongMemEval runners
can drive the [Komplyt Knowledge Graph](https://github.com/Komplyzen/Knowledge-Graph)
as the memory system under test. Spec: `Knowledge-Graph/_bmad-output/specs/spec-public-memory-benchmarks/SPEC.md` (LYT-291).

Everything mem0-specific is untouched. The KG side lives in:

| File | What |
|---|---|
| `benchmarks/common/kg_client.py` | `KgClient` with the `Mem0Client` surface (`add`, `search`, `delete_user`) over MCP JSON-RPC/HTTP |
| `benchmarks/common/kg_extract.py` | LLM extractor turning chat turns into KG nodes; pinned prompt + `PROMPT_VERSION` |
| `scripts/kg_smoke.py` | 3-turn ingest, read-only search, wipe |
| `benchmarks/{locomo,longmemeval}/run.py` | `--backend kg`, `--kg-url`, `--kg-spread`, `--kg-decay`, `--kg-extract-model`; KG fields in result metadata |

## How the mapping works

- One benchmark conversation (`user_id`) = one KG **project** `bench-{user_id}`. Every node create and every query passes the project explicitly, so conversations sharing the tenant never see each other.
- One dataset session (distinct `timestamp`) = one KG **session** (`kg_session start` / `end`), in dataset order. That is what fires `applySessionDecay` between sessions, exactly like production. No timestamp hacks.
- The session date is stored as tag `date:YYYY-MM-DD` and prepended to memory text at search time. It is never put in `content` (it would skew embeddings).
- Search is `kg_query` with `read_only: true` (no implicit activation), `spread_hops: 2`, `limit: min(top_k, 100)`. Ranking handed to the answerer: direct hits by `combined_score`, then spread-only nodes by activation, truncated to `top_k`. `retrieval.query_debug` in each result records `kg_returned` and `kg_after_truncation`.
- Ingest is serialized and the runners force `--max-workers 1` for this backend: KG sessions are tenant-global.
- `delete_user` wipes the project through a **root** SurrealDB connection on the local instance. It refuses `KG_DB_NAME=main`.

## Prerequisites

1. **Local SurrealDB with the migration chain applied**, in a database that is not `main`. From the Knowledge-Graph repo (see its `CLAUDE.md`, "Local SurrealDB for tests"), using e.g. `KG_DB_NAME=bench`.
2. **KG HTTP server** running locally against that database and trusting the **staging** Komplyt issuer, the same way CI does:
   ```bash
   cd Knowledge-Graph/src
   KG_DB_URL=ws://localhost:8000 KG_DB_USER=root KG_DB_PASS=root KG_DB_NS=kg KG_DB_NAME=bench \
   KOMPLYT_ISSUER=https://komplyt-git-staging-komplyzen.vercel.app/api/auth \
   KOMPLYT_JWKS_URL=https://komplyt-git-staging-komplyzen.vercel.app/api/auth/jwks \
   COHERE_API_KEY=... KG_JWT_SECRET=... LISTEN_ADDR=0.0.0.0:3000 bun run index.ts
   ```
   Benchmark data only ever lands in this local database. Staging is used to mint tokens.
3. **A token** for a dedicated benchmark user/tenant on staging: `kg auth login` (writes `~/.kg/credentials.json`), or export `KG_TOKEN`. Tokens expire; a full pass is longer than one token lifetime. Re-run `kg auth login` and `--resume` if you see `401`.
4. **LLM keys**: `OPENAI_API_KEY` for the extractor, answerer and judge.
5. Python deps: `pip install -r requirements.txt`.

Environment used by the `kg` backend:

| Var | Purpose |
|---|---|
| `KG_URL` | KG HTTP server (default `http://localhost:3000`); or `--kg-url` |
| `KG_TOKEN` | Bearer token; overrides `~/.kg/credentials.json` |
| `KG_DB_URL`, `KG_DB_USER`, `KG_DB_PASS`, `KG_DB_NS`, `KG_DB_NAME` | Root SurrealDB access for `delete_user` (HTTP form, e.g. `http://localhost:8000`) |
| `KG_REPO` | Path to the Knowledge-Graph checkout; its HEAD is recorded as `kg_commit` in result metadata |
| `KG_EMBED_PROVIDER` | Free-text label (e.g. `cohere/embed-v4`) recorded in metadata; pin it for the whole comparison |

## Smoke test

```bash
KG_DB_NAME=bench python -m scripts.kg_smoke
```

Ingests three turns, runs the same read-only search twice (ids must match), wipes the project.

## Dev loop: one conversation

Iterate here until per-category numbers stop moving. Same answerer and judge for every system.

```bash
export KG_DB_NAME=bench KG_REPO=../Knowledge-Graph KG_EMBED_PROVIDER=cohere

python -m benchmarks.locomo.run \
  --backend kg --project-name kg-dev \
  --conversations 0 \
  --answerer-model gpt-4o --judge-model gpt-4o \
  --top-k 100 --top-k-cutoffs 10,20,50,100
```

Run it three times (spec CAP-9) and record the spread of overall accuracy. Do the same with `--backend oss` for the mem0 baseline.

## Variants

| Variant | Flags |
|---|---|
| `full` | `--kg-spread on --kg-decay on` (default) |
| `no-spread` | `--kg-spread off --kg-decay on` |
| `no-decay` | `--kg-spread on --kg-decay off` |

The variant name is written to `metadata.kg_variant`. Each variant is a separate ingest (decay is a property of how the data went in), so use a different `--project-name` per variant.

## Full runs

```bash
# LoCoMo, all 10 conversations, categories 1-4
python -m benchmarks.locomo.run --backend kg --project-name kg-full \
  --answerer-model gpt-4o --judge-model gpt-4o --top-k 100 --top-k-cutoffs 10,20,50,100

# LongMemEval S
python -m benchmarks.longmemeval.run --backend kg --project-name kg-full --all-questions \
  --answerer-model gpt-4o --judge-model gpt-4o --top-k 100 --top-k-cutoffs 10,20,50,100

# mem0 OSS baseline (docker compose up -d first), same flags
python -m benchmarks.locomo.run --backend oss --project-name mem0-oss-baseline \
  --answerer-model gpt-4o --judge-model gpt-4o --top-k 100 --top-k-cutoffs 10,20,50,100
```

`--resume` works as for mem0 (ingest checkpoints are per conversation/question). Note that `kg_query` caps `limit` at 100, so cutoff 200 is not available for the KG; mem0's published Cloud number at `top_200` is cited, not reproduced.

## Reading results

Results land in `results/locomo/predicted_<project-name>/` and the unified `locomo_results_<ts>.json`. KG-specific metadata fields: `memory_backend`, `kg_variant`, `kg_spread`, `kg_decay`, `kg_commit`, `kg_extract_model`, `kg_extract_prompt_version`, `kg_extract_prompt_sha256`, `kg_embed_provider`. Per-question `retrieval.search_results[].score_debug` carries `combined_score`, `semantic_score`, `decay_score`, `activation`, `activation_source`.

The results UI (`npm run dev`) browses KG runs next to mem0 runs unchanged.
