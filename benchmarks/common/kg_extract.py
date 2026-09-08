"""
KG Extractor
============

Turns a chunk of chat turns into Komplyt KG nodes. The KG stores deliberate
observations/decisions, not transcripts, so every ``add`` runs an LLM
extraction step first (mem0 does the same inside its own ``add``).

The prompt is pinned here as a module constant. ``PROMPT_VERSION`` and
``prompt_sha256()`` are written into every result file's metadata so a run
can always be tied to the exact extraction prompt that produced it.

Extraction is part of the system under test (spec CAP-3); the model class is
held to mem0 OSS's default (``gpt-4o-mini``) so model capability is not the
variable, only the prompt is.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from benchmarks.common.llm_client import LLMClient

logger = logging.getLogger(__name__)

PROMPT_VERSION = "kg-extract-v1"

# Mirrors NodeType in Knowledge-Graph/src/contracts.ts. Anything else is
# coerced to "observation".
NODE_TYPES = frozenset({
    "observation",
    "decision",
    "argument",
    "todo",
    "problem",
    "mitigation",
    "workaround",
    "approach",
})
DEFAULT_TYPE = "observation"

MAX_CONTENT_CHARS = 600
MAX_NODES_PER_CHUNK = 12
MAX_RELATES_PER_NODE = 20  # kg_node relates_to hard cap (contracts.ts)
PRIOR_CONTEXT_NODES = 30

EXTRACTION_SYSTEM_PROMPT = """You extract durable, self-contained knowledge from a conversation into a personal knowledge graph.

Return a JSON object: {"nodes": [ ... ]}. Each node has:
- "type": one of observation, decision, argument, todo, problem, mitigation, workaround, approach. Use "observation" for facts about people, places, events, preferences and states of the world; "decision" for choices someone made; "todo" for stated plans or intentions; "problem" for difficulties; the rest only when they clearly apply.
- "content": one or two sentences, written in the third person with the speaker's name, standing alone without the conversation. Keep every concrete detail: names, places, numbers, dates, durations, relationships, reasons. Do not paraphrase away specifics. Do not include the date of the conversation itself; it is stored separately.
- "tags": 0 to 5 short lowercase keywords (people, topics, places).
- "relates_to_indices": indices of EXISTING nodes (from the "Existing nodes" list, if given) that this node directly builds on, updates, or contradicts. Empty list if none.

Rules:
- Extract only what is stated or clearly implied. No speculation.
- One fact per node. Split compound statements.
- Skip greetings, pleasantries, and content with no lasting information.
- If a new statement updates or contradicts an existing node, still create the new node and link it via relates_to_indices; never rewrite existing nodes.
- Return {"nodes": []} if nothing is worth keeping.
"""


def prompt_sha256() -> str:
    return hashlib.sha256(EXTRACTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _render_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        # mem0 runners put the speaker name in "name" for LoCoMo, and use
        # plain user/assistant for LongMemEval.
        speaker = m.get("name") or role
        lines.append(f"{speaker}: {m.get('content', '').strip()}")
    return "\n".join(lines)


def _render_prior(prior: list[tuple[int, str]]) -> str:
    if not prior:
        return ""
    lines = [f"[{idx}] {content}" for idx, content in prior]
    return "Existing nodes (index: content):\n" + "\n".join(lines) + "\n\n"


class KgExtractor:
    """LLM extraction of KG nodes from a chunk of turns.

    Args:
        model: Extraction model. Default matches mem0 OSS's fact extractor.
        provider: LLMClient provider ("openai", "anthropic", "azure").
        rpm: LLM requests per minute.
    """

    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai", rpm: int = 200):
        self.model = model
        self.provider = provider
        self.llm = LLMClient(model=model, provider=provider, rpm=rpm)

    async def extract(
        self,
        messages: list[dict[str, str]],
        prior: list[tuple[int, str]] | None = None,
        date_str: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of {type, content, tags, relates_to_indices}.

        ``prior`` is a list of (global_index, content) for nodes already created
        for the same conversation; indices returned in ``relates_to_indices``
        refer to those global indices.
        """
        prior = prior[-PRIOR_CONTEXT_NODES:] if prior else []
        valid_indices = {idx for idx, _ in prior}

        user_prompt = (
            _render_prior(prior)
            + (f"Conversation date: {date_str}\n\n" if date_str else "")
            + "Conversation:\n"
            + _render_messages(messages)
        )

        try:
            raw = await self.llm.generate_structured(system=EXTRACTION_SYSTEM_PROMPT, user=user_prompt)
        except Exception as exc:  # LLMClient already retried
            logger.warning("Extraction failed: %s", str(exc)[:200])
            return []

        nodes_raw = raw.get("nodes", []) if isinstance(raw, dict) else []
        if not isinstance(nodes_raw, list):
            return []

        out: list[dict[str, Any]] = []
        for item in nodes_raw[:MAX_NODES_PER_CHUNK]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            content = content[:MAX_CONTENT_CHARS]

            ntype = str(item.get("type", DEFAULT_TYPE)).strip().lower()
            if ntype not in NODE_TYPES:
                ntype = DEFAULT_TYPE

            tags_raw = item.get("tags", [])
            tags = []
            if isinstance(tags_raw, list):
                for t in tags_raw[:5]:
                    t = str(t).strip().lower()
                    if t:
                        tags.append(t)

            rel_raw = item.get("relates_to_indices", [])
            relates: list[int] = []
            if isinstance(rel_raw, list):
                for r in rel_raw:
                    try:
                        r_int = int(r)
                    except (TypeError, ValueError):
                        continue
                    if r_int in valid_indices and r_int not in relates:
                        relates.append(r_int)
            relates = relates[:MAX_RELATES_PER_NODE]

            out.append({
                "type": ntype,
                "content": content,
                "tags": tags,
                "relates_to_indices": relates,
            })
        return out
