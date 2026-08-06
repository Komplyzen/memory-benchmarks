"""Conductor: a lab notebook with hands for running memory-system evals.

It provides detached run execution around `benchmarks/*/run.py`, records run
state and origin stories in `evals.db`, and tracks reusable artifacts. Approval
and evaluation judgment live in the agent/user workflow; the CLI executes the
approved configuration and records what happened.
"""

__version__ = "0.1.0"
