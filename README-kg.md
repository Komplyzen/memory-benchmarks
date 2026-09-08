# Running the benchmarks against the Komplyt KG

This fork adds two backends to the unmodified LoCoMo and LongMemEval runners:

- `--backend kg`: the [Komplyt Knowledge Graph](https://github.com/Komplyzen/Knowledge-Graph) as the memory system under test, ingested by **Komplyt Zero as shipped**: an LLM agent driving the KG's own MCP tools.
- `--backend none`: no memory system. The answerer gets the whole transcript. This is the floor the small-model KG legs are compared against.

Spec: `Knowledge-Graph/_bmad-output/specs/spec-public-memory-benchmarks/SPEC.md` (LYT-291). Everything mem0-specific is untouched.

| File | What |
|---|---|
| `benchmarks/common/kg_client.py` | `KgClient` with the `Mem0Client` surface (`add`, `search`, `delete_user`) over MCP JSON-RPC/HTTP; guardrails; run metadata |
| `benchmarks/common/kg_agent_ingest.py` | Tool contract from `tools/list`, the agent loop (OpenAI / Azure / Mistral / Anthropic), ingest stats, `ScriptedIngestor` for CI |
| `benchmarks/common/none_client.py` | `NoMemoryClient`: transcript in, transcript out |
| `benchmarks/common/utils.py` | `FULL_CONTEXT_CUTOFF` pseudo-cutoff for `--backend none` |
| `scripts/kg_smoke.py` | One fake session in, read-only search, wipe; `--stub-agent` needs no LLM |
| `benchmarks/{locomo,longmemeval}/run.py` | `--backend kg|none`, `--kg-*`, `--none-context-tokens`; backend fields in result metadata |

## How the KG ingest works

1. **Tool contract.** Once per run the client calls `initialize` and `tools/list` on `/mcp` and converts the tools (name, description, inputSchema) verbatim into provider tool definitions. The server's `instructions` string is appended to the system prompt verbatim. `tool_contract_sha256` (tools + instructions) goes into result metadata; it is what makes two runs comparable.
2. **One agent loop per dataset session.** The runners call `add` per chunk (LoCoMo: one turn; LongMemEval: one user/assistant pair). The client buffers chunks until the `(user_id, timestamp)` key changes, then hands the whole session to the agent as one user message: `Session dated {date} between {a} and {b}:` followed by the turns. The system prompt is static (fixed framing + server instructions) so provider prompt caching covers tools + system. The model decides which tools to call (`kg_session`, `kg_query`, `kg_node`, `kg_update`, ...); the harness executes each call against `/mcp` and returns the result text. The loop ends when the model stops calling tools or hits `--kg-max-tool-calls` (default 12, logged and counted).
3. **Guardrails the harness enforces, not the model.** Every `kg_node` create gets `project=bench-{user_id}` and a `date:YYYY-MM-DD` tag (the date never goes into `content`). A write without an active session first gets a harness-started session. A `kg_session start` from the model is deduplicated against the session already open for this dataset session and forced onto the benchmark project. If the model does not end the session, the harness ends it when the next dataset session starts (or at search/close), so `applySessionDecay` fires exactly once per dataset session. `--kg-decay off` keeps one session for the whole run: model start/end calls are acknowledged, not executed.
4. **Search** is unchanged: one `kg_query` with `read_only: true`, `spread` per variant, `spread_hops: 2`, `limit: min(top_k, 100)`, scoped to the project. Direct hits ranked by `combined_score`, then spread nodes by activation, truncated to `top_k`; `retrieval.query_debug` records `kg_returned` / `kg_after_truncation`. The date tag is prepended to memory text at answer time.
5. **Isolation.** KG sessions are tenant-global, so ingest is serialized and the runners force `--max-workers 1` for this backend. Search runs concurrently. `delete_user` wipes the project through a **root** SurrealDB connection on the local instance and refuses `KG_DB_NAME=main`.

### Agent models and providers

| Profile | Flags |
|---|---|
| small-model | `--kg-agent-provider anthropic --kg-agent-model claude-haiku-4-5` |
| head-to-head | `--kg-agent-provider openai --kg-agent-model gpt-5` |
| default | `--kg-agent-provider openai --kg-agent-model gpt-5-mini` |
| Mistral | `--kg-agent-provider mistral --kg-agent-model mistral-small-latest` (OpenAI-compatible endpoint, `MISTRAL_API_KEY`) |

Reasoning is `minimal` for OpenAI reasoning models and thinking is off for Anthropic. Provider and exact model id are recorded as `kg_agent_provider` / `kg_agent_model`.

Prompt caching: OpenAI caches the stable prefix automatically (tools are serialized once, in sorted order). Anthropic needs explicit `cache_control` breakpoints, which are set on the last tool and on the system block; note Anthropic only caches prefixes above its minimum size (1024 tokens, 2048 for Haiku), so short contracts may show a zero hit ratio. Mistral reports no cache fields. Metadata records `kg_ingest_cached_prompt_tokens`, `kg_ingest_cache_write_tokens`, `kg_ingest_cache_hit_ratio`, and the ingest summary line printed at the end of a run includes the hit ratio.

### The mem0 OSS baseline uses the same extraction model

The mem0 OSS server reads its extraction model once at container start (`LLM_MODEL` in `docker-compose.yml`, or a mounted `config.yaml`); the harness cannot change it per run. Start the server with the model you use for the KG agent so both ingests use the same model class:

```bash
LLM_MODEL=gpt-5-mini docker compose up -d --force-recreate
```

`--kg-agent-model` on an `--backend oss` run is recorded in metadata but does not reconfigure the server.

## `--backend none`: the full-context floor

No memory system. `add` stores the transcript per conversation; `search` returns it as one memory with `=== Session dated ... ===` headers, so the answerer sees the full conversation plus the question and is judged as usual. Metrics are reported at a single pseudo-cutoff `full_context`. LoCoMo conversations fit (about 9k tokens). LongMemEval histories may not: the transcript is truncated from the oldest session until it fits `--none-context-tokens` (default 150k, chars/4 approximation); `query_debug` and metadata record how many sessions and characters were dropped.

## Prerequisites

1. **Local SurrealDB with the migration chain applied**, in a database that is not `main` (see the Knowledge-Graph `CLAUDE.md`, "Local SurrealDB for tests"), e.g. `KG_DB_NAME=bench`.
2. **KG HTTP server** running locally against that database and trusting the **staging** Komplyt issuer, the same way CI does:
   ```bash
   cd Knowledge-Graph/src
   KG_DB_URL=ws://localhost:8000 KG_DB_USER=root KG_DB_PASS=root KG_DB_NS=kg KG_DB_NAME=bench \
   KOMPLYT_ISSUER=https://komplyt-git-staging-komplyzen.vercel.app/api/auth \
   KOMPLYT_JWKS_URL=https://komplyt-git-staging-komplyzen.vercel.app/api/auth/jwks \
   COHERE_API_KEY=... KG_JWT_SECRET=... LISTEN_ADDR=0.0.0.0:3000 bun run index.ts
   ```
   Benchmark data only ever lands in this local database. Staging is used to mint tokens.
3. **A token** for a dedicated benchmark user/tenant on staging: `kg auth login` (writes `~/.kg/credentials.json`), or export `KG_TOKEN`. Tokens expire; a full pass is longer than one token lifetime. Re-run `kg auth login` and `--resume` on `401`.
4. **LLM keys**: `OPENAI_API_KEY` (answerer, judge, default agent), plus `ANTHROPIC_API_KEY` or `MISTRAL_API_KEY` for those agent providers.
5. Python deps: `pip install -r requirements.txt`.

Environment used by the `kg` backend:

| Var | Purpose |
|---|---|
| `KG_URL` | KG HTTP server (default `http://localhost:3000`); or `--kg-url` |
| `KG_TOKEN` | Bearer token; overrides `~/.kg/credentials.json` |
| `KG_DB_URL`, `KG_DB_USER`, `KG_DB_PASS`, `KG_DB_NS`, `KG_DB_NAME` | Root SurrealDB access for `delete_user` (HTTP form, e.g. `http://localhost:8000`) |
| `KG_REPO` | Path to the Knowledge-Graph checkout; its HEAD is recorded as `kg_commit` |
| `KG_EMBED_PROVIDER` | Free-text label (e.g. `cohere/embed-v4`) recorded in metadata; pin it for the whole comparison |

## Smoke test

```bash
KG_DB_NAME=bench python -m scripts.kg_smoke                 # real agent (gpt-5-mini)
KG_DB_NAME=bench python -m scripts.kg_smoke --stub-agent    # no LLM; for CI
```

`--stub-agent` runs a fixed sequence through the same guardrails: session start, one `kg_query`, three `kg_node` creates (one with `relates_to`), one `kg_update`, session end. Both variants then search twice read-only (ids must match) and wipe the project.

## Dev loop: one conversation

```bash
export KG_DB_NAME=bench KG_REPO=../Knowledge-Graph KG_EMBED_PROVIDER=cohere

python -m benchmarks.locomo.run \
  --backend kg --project-name kg-dev --conversations 0 \
  --kg-agent-provider openai --kg-agent-model gpt-5-mini \
  --top-k 100 --top-k-cutoffs 20,100
```

Run it three times (spec CAP-9) and record the spread of overall accuracy. Do the same with `--backend oss` and `--backend none`. The answerer and judge defaults are the upstream ones (`gpt-5`); pass `--answerer-model` / `--judge-model` identically to every system in a comparison.

## Variants

| Variant | Flags |
|---|---|
| `full` | `--kg-spread on --kg-decay on` (default) |
| `no-spread` | `--kg-spread off --kg-decay on` |
| `no-decay` | `--kg-spread on --kg-decay off` |

Written to `metadata.kg_variant`. Each variant is a separate ingest, so use a different `--project-name` per variant.

## Cost

Rough figures at cutoffs 20 and 100, gpt-5 answerer and judge:

| Leg | Approx. cost |
|---|---|
| LoCoMo, one KG variant (agent ingest + answer + judge) | €25 to 40 |
| LongMemEval S, one KG variant, agent ingest | about €200 |
| LongMemEval S, extraction-style ingest (mem0 OSS baseline) | about €36 |
| `--backend none` | answer + judge only, but each answer carries the full transcript |

The agent ingest is the expensive part: one LLM call per tool-calling turn per dataset session, with the tool contract in every prompt. Prompt caching cuts that where the provider supports it. Iterate on single conversations first; run a full pass only when per-category numbers stop moving.

## Full runs

```bash
# LoCoMo, all 10 conversations, categories 1-4
python -m benchmarks.locomo.run --backend kg --project-name kg-full \
  --top-k 100 --top-k-cutoffs 20,100

# LongMemEval S
python -m benchmarks.longmemeval.run --backend kg --project-name kg-full --all-questions \
  --top-k 100 --top-k-cutoffs 20,100

# floor
python -m benchmarks.locomo.run --backend none --project-name none-full

# mem0 OSS baseline (LLM_MODEL=<agent model> docker compose up -d first), same flags
python -m benchmarks.locomo.run --backend oss --project-name mem0-oss-baseline \
  --top-k 100 --top-k-cutoffs 20,100
```

`--resume` works as for mem0 (ingest checkpoints are per conversation/question). Because chunks are buffered into whole dataset sessions before the agent runs, a run interrupted mid-session loses that session's buffered turns on resume (the runner's chunk checkpoint considers them done). `kg_query` caps `limit` at 100, so cutoff 200 is not available for the KG; mem0's published Cloud number at `top_200` is cited, not reproduced.

## CI / bench.sh interface

`.github/scripts/bench.sh` in the Knowledge-Graph repo (spec-benchmark-ci) drives these fork-side scripts. Backend names everywhere: `kg-full`, `kg-no-spread`, `kg-no-decay`, `mem0-oss`, `none` (`no-memory` is accepted as an alias and normalized to `none`). Flags per backend live in `benchmarks/common/bench_common.py` (`BACKEND_FLAGS`).

**Profiles and prices** `benchmarks/common/profiles.json`: `cheap` (gpt-5-nano everywhere, judge gpt-5), `head-to-head` (all gpt-5), `small-model` (claude-haiku-4-5 answerer/agent/mem0 extraction, judge gpt-5, adds the `none` leg). `prices` are list prices per 1M tokens (USD) with `prices_recorded_on`; verify before quoting. mem0 OSS's default config is OpenAI-only; the small-model profile's `mem0_llm_model` needs an Anthropic-capable mem0 config.

| Script | Purpose | Exit codes |
|---|---|---|
| `scripts/gate_estimate.py --benchmark locomo\|longmemeval --shards CSV\|all [--shard-size 25] --cutoffs CSV --backends CSV --profile NAME [--answerer-model ...] [--dataset-path F] [--expected-sha256 HEX] [--max-questions N] [--full]` | Validate inputs, download + sha256 the dataset (runner's own download function), resolve shards, estimate calls and EUR. **One JSON object on stdout**: `{benchmark, profile, models, cutoffs, backends, shards:[{idx, questions, sessions, dataset_path}], totals:{questions, sessions, calls, eur}, per_backend:{name:{calls, eur}}, dataset_sha256, prices_recorded_on, price_missing, full, ceiling}`. Prose on stderr. For longmemeval `shards[].dataset_path` is the slice file *name* (`longmemeval_s_shard<idx>.json`); `slice_dataset.py` decides the directory. | 2 invalid input (unknown backend named, cutoff > 100 for kg-*, bad shard, unknown profile), 3 sha256 mismatch (both printed), 4 calls > 2500 without `--full` (JSON still printed), 5 `KG_EMBED_FIXTURES` set |
| `scripts/slice_dataset.py --benchmark longmemeval --shard-size 25 --out-dir DIR [--dataset-path F]` | Writes `longmemeval_s_shard<idx>.json` blocks; prints `[{idx, path, questions}]`. For locomo prints `[]` (shards are conversations). | 2 bad input |
| `scripts/merge_results.py --benchmark B --backend NAME --expected-shards CSV [--allow-partial] --out FILE --run-id ID --run-attempt N DIR...` | Merge legs of one backend into the runner's unified shape; metrics recomputed with `compute_locomo_metrics` / `compute_longmemeval_metrics`. | 2 zero evaluations, backend mismatch, missing shards (unless `--allow-partial`, then `missing_shards` listed), legs differing in answerer_model / judge_model / cutoffs / harness_sha / kg_variant, duplicate question_id |
| `scripts/strip_for_publish.py IN OUT [--keep-questions]` | Remove dataset text before publishing: `retrieval.search_results[].memory`, `user_profile`, and (unless `--keep-questions`) `question`, `ground_truth_answer`, `retrieval.search_query`. Ids, scores, `score_debug`, `evidence` (dialog ids), model output kept. | |
| `scripts/summary_table.py --out SUMMARY.md [--full] [--failed backend:shard:phase ...] MERGED.json...` | Job-summary markdown: header (run id, harness SHA, KG commit, models, cutoffs, provenance lines, dev-loop notice with shards unless `--full`), one row per backend, delta rows vs `mem0-oss`, one line per failed leg. | |

**Leg directory** = the runner's `results/<benchmark>/predicted_<project>/`: per-question `<question_id>.json` files, runner checkpoints prefixed `_`, plus a sidecar **`leg_meta.json`** that bench.sh writes:

```json
{
  "backend": "kg-full", "shard": 0,
  "models": {"answerer_model": "...", "answerer_provider": "...", "judge_model": "...", "judge_provider": "...",
             "agent_model": "...", "agent_provider": "...", "mem0_llm_model": "..."},
  "cutoffs": [20, 100],
  "harness_sha": "<fork commit the leg ran>",
  "kg_commit": "<KG commit>",
  "kg_meta": { "...client.ingest_metadata() of the leg (kg-* legs)" },
  "wall_seconds": 1234.5,
  "phase_failed": "ingest",
  "embed_failures": 0,
  "mem0": {"llm_model": "gpt-5-nano", "search_flags": {"rerank": false, "top_k": 100}, "image_digest": "sha256:..."},
  "versions": {"docker": "...", "bun": "...", "python": "..."}
}
```

Required: `backend`, `shard`, `models`, `cutoffs`, `harness_sha`. Everything else is optional and feeds the summary (wall time, nodes per shard from `kg_meta.kg_nodes_created`, embed failures, mem0 flags). The runner itself writes its unified file to `--output-dir`, not into the predicted directory, so `leg_meta.json` is the only metadata merge reads per leg.

**Merged file** (`<benchmark>_<backend>_results.json`): `{metadata, metrics_by_cutoff, evaluations}` like the runner's own, with metadata adding `backend`, `kg_variant`, `github_run_id`, `run_attempt`, `harness_sha` (git rev-parse HEAD of the fork at merge time; `leg_harness_sha` is what the legs recorded), `kg_commit`, `embedding_model`, `legs: [{shard, dir, evaluations, wall_seconds, phase_failed, kg_ingest_llm_calls, answer_judge_calls, nodes_created, embed_failures}]`, `expected_shards`, `missing_shards`, `wall_seconds_total`, `answer_judge_calls`, summed `kg_ingest_*` counters, `mem0_llm_model`, `mem0_search_flags`, `mem0_image_digest`.

**KG_CI mode.** With `KG_CI=1` the client reports a 401 as "token or JWKS server" and retries once before failing, instead of advising `kg auth login`.

## Reading results

Results land in `results/locomo/predicted_<project-name>/` and the unified `locomo_results_<ts>.json`. Backend metadata: `memory_backend`, `kg_variant`, `kg_spread`, `kg_decay`, `kg_commit`, `kg_ingest_mode`, `kg_agent_provider`, `kg_agent_model`, `tool_contract_sha256`, `kg_ingest_*` (sessions, LLM calls, prompt/completion/cached tokens, cache hit ratio, tool calls total and by name, tool calls per session, tool calls per LLM call and its histogram, `max_tool_calls_hit`), `kg_guardrails` (how often the harness had to start/end/dedupe sessions or inject the project), `kg_embed_provider`; for `none`: `none_context_token_budget`, `none_questions_truncated`, `none_sessions_dropped_total`. Per-question `retrieval.search_results[].score_debug` carries `combined_score`, `semantic_score`, `decay_score`, `activation`, `activation_source`.

The results UI (`npm run dev`) browses KG and none runs next to mem0 runs unchanged.
