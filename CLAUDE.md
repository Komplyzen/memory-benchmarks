# memory-benchmarks — Agent Operator Manual

This repo contains memory-system benchmarks (LongMemEval, LOCOMO, BEAM,
BrowseComp-Plus) and
**conductor** — a CLI that runs them, records them, and reuses expensive
intermediate artifacts. If you are an agent asked to "run evals", this file is
your operating procedure. Internal deployment specifics live in `CLAUDE.local.md`
(gitignored); read it if present.

## The mental model (do not skip)

Every eval is a ladder of artifacts, each made from the one above:

    dataset → extracted memories → embedded store → retrieval results → answers → judgments
              (hours, $$$)         (minutes)        (seconds)          (cheap)   (cheap)

A change enters at some rung; everything ABOVE it is reusable by construction.
New judge prompt → rung 5: re-judge existing answers. New top_k/reranker →
rung 4: re-search an existing store. New embedder → rung 3: re-embed the shelved
extracted text. New extraction model or extraction-touching platform change →
rung 2: pay full extraction (only for the changed side; baselines are usually
already on the shelf).

Division of labor:
- **The machine (conductor)** executes, records, and keeps artifacts. It is
  deliberately dumb: no approval gates, no config schema, no judgment.
- **The agent (you)** decides what to run, what is reusable, and interprets
  results — always with the user's visibility/approval BEFORE invoking commands.
  Approval lives in the agent↔user layer only; never expect the CLI to gate you.

## The measurement contract

Before setting up or launching any evaluation, state in plain English:

1. The exact question or decision the evaluation is meant to inform.
2. The concrete system under test and baseline, including target, models,
   embedder, extraction path, search mode, and intentionally varied dimensions.
3. The dataset revision and evaluated slice, inclusion/exclusion rules, and
   available ground truth.
4. Which ladder stages will actually run and which will be reused, cached,
   mocked, or skipped, including the origin of every reused artifact.
5. What every metric counts, its denominator and unit, failure handling, and
   whether it is canonical or directional.
6. What the result can support, what it does not measure or imply, and the
   remaining confounders.

The user approves this measurement contract, not just the shell command. If the
setup changes, revise the contract and obtain approval again. Final reports must
lead with: what ran, what did not run, what was measured, and what the result
supports and does not establish.

## Iron rules

1. **Never substitute the system under test.** The embedder / extraction model /
   memory platform being benchmarked is not swappable for a stand-in, even "just
   to verify mechanics". If it's down or creds expired: STOP, tell the user, wait.
2. **Fail loud, never silently degrade.** Preflight failures, credential expiry,
   rate-limit storms → halt and surface. An eval that silently substitutes or
   drops work is worse than no eval.
3. **Faithfulness before reuse.** Reused artifacts must have been produced by the
   real pipeline (conductor's replay runs the actual ingestion path with only the
   recorded extraction substituted). If you invent a new reuse seam, audit EVERY
   side effect of the normal path (payload fields, entity store, dedup, history)
   before trusting it.
4. **Record everything.** Every run gets an origin story (auto-captured) and your
   freeform notes (`--note`, `--describe k=v`). Every reuse decision gets a
   rationale. If you reused something "close enough" for a directional read, SAY
   SO in the note — visibility makes it legitimate.
5. **Public repo hygiene.** Never commit customer data, internal endpoints,
   account ids, credentials, or run notes. `conductor_state/`, `.env`,
   `CLAUDE.local.md`, `EXPECTED_BEHAVIOR.local.md`, and
   `BENCHMARK_PROFILES.local.md` are gitignored.
6. **No number without semantics.** Report the evaluated population,
   denominator, units, failure handling, mode, and claim boundary with every
   metric. A metric name alone is not an interpretation.

## The CLI contract

    conductor start <benchmark> [--reuse STORE] [--set k=v ...] [--env K=V ...]
                    [--project-name P] [--note "..."] [--describe k=v ...]
      → prints run id, returns immediately; run is detached (survives your session).
        --set keys map to the benchmark run.py flags (snake_case → --kebab-case).
    conductor ls [-n N] [--status s]     # ledger: runs, status, headline accuracy
    conductor status <run>               # full record: origin, git state, config, metrics
    conductor logs <run> [-n N]          # tail the run log
    conductor stop <run>
    conductor diff <runA> <runB>         # config delta, shared store, per-cutoff metric deltas
    conductor shelf ls|show|adopt        # artifact shelf (extracted_memories, memory_store)
    conductor materialize <artifact> --host H [--store-id S] [--question-ids a,b]
      → replays extracted memories through the REAL pipeline into a live, faithful
        store (entity linking, dedup, BM25 all real). Resumable: same --store-id
        continues from its checkpoint. On embedder/cred failure it halts loudly.
    conductor score <run> --scorer <path>
      → run any scorer over a run's artifacts (contract: conductor/score.py).
        Results land MODE-TAGGED with validity notes in the ledger; directional
        metrics never masquerade as canonical. Scorers are derived by the
        eval-operator per benchmark profile, approved by the human, and saved
        under scorers/<benchmark>/ when reusable and safe to share.
    conductor verify-store <store> [--sample N]
      → instruments, EVIDENCE NOT VERDICT: timestamps honored, entity store
        populated, counts match manifest, search returns. Red flags (exit 1) =
        strong evidence of a broken setup. All-clear ≠ sound — the failure space
        is open; the operator's open-ended review decides, and its signed
        rationale (--describe operator_review=...) goes in the ledger.

For any eval setup/launch/wiring task, dispatch the `eval-operator` subagent
(.claude/agents/eval-operator.md): it elicits experiment intent, gathers
instrument evidence, reviews against expected target behavior, and signs off
in writing before launching. Optional `.local.md` files provide machine-specific
context when present; their absence is never a reason to guess. Knowledge grows
by agent-proposes / human-approves.

A `memory_store` artifact + `--reuse <store>` = a run that skips ingestion
entirely (conductor pre-writes the runner's ingestion checkpoints; the runner
adopts the store's user_ids and goes straight to search→answer→judge).

## Standard procedures

### Onboard a custom benchmark or customer dataset
1. Read the data and existing harness. State the capability being tested, the
   unit of evaluation, and every available ground-truth field.
2. Define the ladder for this evaluation (ingest, store, retrieve, answer,
   judge, or the relevant subset) and which dimensions must remain fixed.
3. Select the cheapest faithful mode. If a new scorer is needed, implement the
   `conductor/score.py` contract and state its validity and fairness conditions.
4. Add `benchmarks/<name>/run.py`. It must accept `--project-name`, write a
   durable result artifact, and print `Results saved to: <path>` on completion.
5. Register it in `conductor/launcher.py` and `conductor/db.py`. Add a canonical
   metric extractor in `conductor/metrics.py` only if a compact headline metric
   preserves the benchmark's meaning; derived scorers need no extractor.
6. Add a deterministic unit test, run a small real-path smoke test, present the
   operator review and expected cost/time, and obtain approval before scaling.

Customer datasets, endpoints, credentials, run artifacts, and private expected
behavior remain local. Reusable adapters and scorers may be committed only when
they contain no customer material.

### Run an experiment on an existing store (most common)
1. `conductor shelf ls` — find the memory_store; `shelf show` its origin (which
   platform build, embedder, extraction model produced it).
2. Decide reusability for THIS experiment (see reuse procedure below). Present
   plan + cost to the user, get approval.
3. `conductor start longmemeval --reuse <store> --set all_questions=true
   --set provider=azure --set answerer_model=<m> --set judge_model=<m>
   --set max_workers=4 --set rpm=15 --note "<what and why>"`
4. Watch `logs` for a couple minutes: you want "already ingested" lines (reuse
   working) and near-zero 429s (throttle right). Then leave it; it's detached.
5. On completion, metrics land in the ledger automatically. Check `status` for
   an INTEGRITY_WARNING (empty answers = LLM retries exhausted = biased metrics;
   if present, quarantine those question results and rerun them).
6. `conductor diff <baseline> <this-run>` to report the comparison.

### Reuse decision procedure
Reusable iff the candidate artifact's origin matches the current experiment in
every dimension that FEEDS the artifact's rung:
- extracted memories: same dataset + extraction model/prompt + platform
  extraction code. Retrieval knobs are irrelevant to this rung.
- memory_store: all of the above + same embedder + same platform search-side
  state you intend to test against.
Check origin stories (`shelf show`, `status`) against the current target. For a
platform BRANCH: diff the branch against the platform commit recorded in the
artifact's origin; if the diff touches extraction paths, rung 2 is invalidated;
if only retrieval/search paths, stores may still stand for directional runs
(label it). When uncertain: ask the user, propose the conservative option.

### New platform branch end-to-end
1. Stand up the branch as a running stack (worktree + its own compose/stack;
   see CLAUDE.local.md for the concrete stack recipe on this machine).
2. Preflight it (`conductor start` does this automatically; a dead/credless
   target halts before anything is recorded).
3. Make the reuse call vs the shelf (procedure above), get user approval.
4. Materialize if needed (`--store-id <branch-tag>`), then run + diff vs baseline.

### Wire a new model (embedder / extraction LLM)
There is NO uniform way — hosting providers differ every time. The pattern:
1. Agent writes an adapter file in `docker/mem0/` (see `sagemaker_embedder.py`,
   `kimi_llm.py` as templates) + a `mem0-config-<name>.yaml` + (if needed) a
   compose service with the right env passthrough.
2. **Preflight is the definition of done** — the wiring isn't complete until the
   probe answers through the real path.
3. The origin story records endpoint + config, so results stay traceable.
Bespoke once, mechanical forever. Do not attempt to schema-ize providers.

## Known failure modes (learned the hard way — do not relearn)

- **LLM rate-limit storms poison results**: the shared llm_client returns "" when
  its 5 retries exhaust; empty answers get scored WRONG silently. Throttle until
  429s are rare (they're absorbed by backoff). conductor stamps INTEGRITY_WARNING
  with the empty-answer count — treat any nonzero as "rerun those questions".
- **More parallelism can be slower**: a single embedding endpoint saturates at
  ~3-4 concurrent. Measured: raising server workers 5→9 made materialize ~1.9x
  SLOWER. Measure, don't assume. Server workers default 5 (UVICORN_WORKERS).
- **Temporary cloud creds expire mid-run**: materialize checkpoints per question
  and halts loudly; refresh creds, then `docker compose up -d --force-recreate`
  the mem0 service (it reads creds at container start), then re-run the same
  materialize command — it resumes. Runner-side runs resume via --project-name.
- **Ingestion faithfulness**: never ingest shelved memories with `infer=false`
  directly — it skips entity linking + dedup (measured: 0 entities vs 8664 on
  the same data). Always go through `/replay` / `/replay_bulk` (materialize does).
- **Optimize only after end-to-end profiling**: embed batching microbenchmarks
  promised ~11x; the real pipeline got 1.39x because CPU-bound per-event work
  dominated. Amdahl before building.

## Layout

- `benchmarks/<name>/run.py` — pipelines (ingest/search/answer/judge phases).
- `conductor/` — the CLI (db, launcher, supervisor, shelf, materialize, reuse,
  preflight, metrics, record).
- `evals.db` — shared ledger (web UI reads the same file). `conductor_state/` —
  runs, shelf artifacts, logs (gitignored).
- `docker/mem0/` — OSS mem0 server + model adapters. `docker-compose.yml` —
  per-model services + qdrant.
- The web UI reads the same `evals.db` run ledger. CLI-only use does not require
  the dashboard.
