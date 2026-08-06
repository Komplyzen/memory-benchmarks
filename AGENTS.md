# Agent instructions

## Memory evaluation workflow

- For any task involving memory benchmark design, custom or customer data,
  Mem0 evaluation, scorer design, evaluation setup, artifact reuse, or an eval
  launch, first read `CLAUDE.md` and `.claude/agents/eval-operator.md`.
- Treat the eval operator as the judgment layer: establish the experiment's
  intent, ground truth, validity conditions, fixed dimensions, and approval.
- Use `conductor` as the execution and record-keeping layer when the task needs
  to launch, monitor, score, compare, materialize, or reuse an evaluation.
- Files ending in `.local.md` contain optional machine or deployment context.
  Read them when present, but do not assume teammates have them and never
  commit their contents.
- Before launching paid or long-running work, present the exact dataset,
  system under test, reusable artifacts, evaluation mode, expected cost/time,
  and operator review to the user for approval.
- Never substitute the system under test or silently fall back to another
  model, embedder, endpoint, dataset, or scoring mode.
