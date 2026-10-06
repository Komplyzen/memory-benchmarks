"""
KG Agent Ingest
===============

Ingest for the ``kg`` backend as "Komplyt Zero as shipped": an LLM agent with
the KG's own MCP tools decides what to keep. The harness does not script the
tool calls; it only

  1. fetches the server's tool list and instructions once (``initialize`` +
     ``tools/list``) and converts them verbatim into provider tool definitions
     (OpenAI function calling or Anthropic tool use), hashing the contract for
     result metadata,
  2. runs a tool-calling loop per dataset session with a minimal, pinned
     system prompt plus the server instructions,
  3. executes each tool call through a callback supplied by ``KgClient``,
     which applies the guardrails (project injection, date tag, session
     bracketing) that keep conversations isolated and decay comparable.

Providers: ``openai`` / ``azure`` (chat.completions with ``tools``) and
``anthropic`` (messages with ``tools``, tool_use / tool_result blocks).
Reasoning is set to minimal for OpenAI reasoning models and thinking is left
off for Anthropic. The existing ``LLMClient`` stays in charge of answerer and judge.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# OpenAI rejects function descriptions longer than this.
OPENAI_TOOL_DESCRIPTION_MAX = 1024
ANTHROPIC_MAX_TOKENS = 4096

# Static across sessions so provider prompt caching covers tools + system.
# Speakers and the date go into the first line of the user message instead.
SYSTEM_PROMPT = (
    "You are the user's assistant with the Komplyt memory connector. "
    "You are given one session of a conversation between two people. "
    "Use your tools as you normally would to keep what matters for later."
)
USER_HEADER_TEMPLATE = "Session dated {date} between {speaker_a} and {speaker_b}:"

# OpenAI-compatible endpoints selectable via --kg-agent-provider.
OPENAI_COMPATIBLE = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY"},
    # Any OpenAI-compatible endpoint (vLLM, Parasail, Aleph Alpha, ...): url and key from env,
    # matching the `openai-compatible` entry in profiles.json's provider registry.
    "openai-compatible": {"base_url": os.getenv("LLM_BASE_URL"), "key_env": "LLM_API_KEY"},
}

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


# ---------------------------------------------------------------------------
# Tool contract
# ---------------------------------------------------------------------------


@dataclass
class McpToolContract:
    """The server's tools and instructions, as the agent sees them."""

    raw_tools: list[dict[str, Any]]
    instructions: str
    openai_tools: list[dict[str, Any]]
    anthropic_tools: list[dict[str, Any]]  # last tool carries cache_control
    sha256: str
    truncated_descriptions: list[str] = field(default_factory=list)

    @property
    def system_text(self) -> str:
        """Static system prompt: fixed framing + server instructions verbatim."""
        return f"{SYSTEM_PROMPT}\n\n{self.instructions}" if self.instructions else SYSTEM_PROMPT

    @property
    def anthropic_system(self) -> list[dict[str, Any]]:
        return [{"type": "text", "text": self.system_text, "cache_control": {"type": "ephemeral"}}]


def contract_from_mcp(tools: list[dict[str, Any]], instructions: str | None) -> McpToolContract:
    """Convert an MCP ``tools/list`` result into provider tool definitions.

    The hash covers the verbatim names, descriptions, input schemas and
    instructions. Truncation to OpenAI's 1024-char description limit happens
    only on the OpenAI copy and is recorded in ``truncated_descriptions``.
    """
    instructions = instructions or ""
    hasher = hashlib.sha256()
    openai_tools: list[dict[str, Any]] = []
    anthropic_tools: list[dict[str, Any]] = []
    truncated: list[str] = []

    for t in sorted(tools, key=lambda x: x.get("name", "")):
        name = t.get("name", "")
        description = t.get("description", "") or ""
        schema = t.get("inputSchema") or {"type": "object", "properties": {}}
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(description.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
        hasher.update(b"\x01")

        sent_description = description
        if len(sent_description) > OPENAI_TOOL_DESCRIPTION_MAX:
            sent_description = sent_description[: OPENAI_TOOL_DESCRIPTION_MAX - 1] + "…"
            truncated.append(name)

        openai_tools.append({
            "type": "function",
            "function": {"name": name, "description": sent_description, "parameters": schema},
        })
        anthropic_tools.append({"name": name, "description": description, "input_schema": schema})

    hasher.update(b"\x02")
    hasher.update(instructions.encode("utf-8"))

    if anthropic_tools:
        # Cache breakpoint after the (deterministically ordered) tool block.
        anthropic_tools[-1] = {**anthropic_tools[-1], "cache_control": {"type": "ephemeral"}}

    if truncated:
        logger.warning("Tool descriptions truncated to %d chars for OpenAI: %s", OPENAI_TOOL_DESCRIPTION_MAX, truncated)

    return McpToolContract(
        raw_tools=tools,
        instructions=instructions,
        openai_tools=openai_tools,
        anthropic_tools=anthropic_tools,
        sha256=hasher.hexdigest(),
        truncated_descriptions=truncated,
    )


# ---------------------------------------------------------------------------
# Speaker / message helpers
# ---------------------------------------------------------------------------

_SPEAKER_PREFIX = re.compile(r"^([A-Za-z][\w .'\-]{0,40}?):\s")


def infer_speakers(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """LoCoMo chunks carry "Name: text" in content; LongMemEval has plain roles."""
    names: dict[str, str] = {}
    for m in messages:
        role = m.get("role", "user")
        if role in names:
            continue
        if m.get("name"):
            names[role] = str(m["name"])
            continue
        match = _SPEAKER_PREFIX.match(m.get("content", ""))
        if match:
            names[role] = match.group(1)
    return names.get("user", "the user"), names.get("assistant", "the assistant")


def render_session(messages: list[dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        content = m.get("content", "").strip()
        if _SPEAKER_PREFIX.match(content):
            lines.append(content)
        elif m.get("name"):
            lines.append(f"{m['name']}: {content}")
        else:
            lines.append(f"{m.get('role', 'user')}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    sessions: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0  # cache read hits
    cache_write_tokens: int = 0    # Anthropic cache_creation_input_tokens
    tool_calls_total: int = 0
    tool_calls_by_name: Counter = field(default_factory=Counter)
    tool_errors_by_name: Counter = field(default_factory=Counter)
    tool_calls_per_session: list[int] = field(default_factory=list)
    tool_calls_per_llm_call: Counter = field(default_factory=Counter)  # n tool calls -> responses
    max_tool_calls_hit: int = 0
    llm_failures: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        return round(self.cached_prompt_tokens / self.prompt_tokens, 4) if self.prompt_tokens else 0.0

    def summary_line(self) -> str:
        return (
            f"kg ingest: {self.sessions} sessions, {self.llm_calls} LLM calls, "
            f"{self.tool_calls_total} tool calls, prompt={self.prompt_tokens} "
            f"(cached {self.cached_prompt_tokens}, hit ratio {self.cache_hit_ratio:.0%}, "
            f"cache writes {self.cache_write_tokens}), completion={self.completion_tokens}, "
            f"max_tool_calls hit {self.max_tool_calls_hit}x"
        )

    def as_metadata(self) -> dict[str, Any]:
        per = self.tool_calls_per_session
        fanout_total = sum(n * c for n, c in self.tool_calls_per_llm_call.items())
        fanout_calls = sum(self.tool_calls_per_llm_call.values())
        return {
            "kg_ingest_sessions": self.sessions,
            "kg_ingest_llm_calls": self.llm_calls,
            "kg_ingest_prompt_tokens": self.prompt_tokens,
            "kg_ingest_completion_tokens": self.completion_tokens,
            "kg_ingest_cached_prompt_tokens": self.cached_prompt_tokens,
            "kg_ingest_cache_write_tokens": self.cache_write_tokens,
            "kg_ingest_cache_hit_ratio": self.cache_hit_ratio,
            "kg_ingest_tool_calls_total": self.tool_calls_total,
            "kg_ingest_tool_calls_by_name": dict(self.tool_calls_by_name),
            "kg_ingest_tool_errors_by_name": dict(self.tool_errors_by_name),
            "kg_ingest_tool_calls_per_session": {
                "min": min(per) if per else 0,
                "max": max(per) if per else 0,
                "mean": round(sum(per) / len(per), 2) if per else 0,
            },
            "kg_ingest_tool_calls_per_llm_call": round(fanout_total / fanout_calls, 2) if fanout_calls else 0,
            "kg_ingest_tool_calls_per_llm_call_histogram": {str(k): v for k, v in sorted(self.tool_calls_per_llm_call.items())},
            "kg_ingest_max_tool_calls_hit": self.max_tool_calls_hit,
            "kg_ingest_llm_failures": self.llm_failures,
        }


@dataclass
class _ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string (OpenAI) or json.dumps(input) (Anthropic)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def _is_openai_reasoning_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


class AgentIngestor:
    """Tool-calling loop over the KG's MCP tools.

    Args:
        model: Agent model id, recorded verbatim in metadata. Default ``gpt-5-mini``.
        provider: "openai", "azure" or "anthropic".
        max_tool_calls: Per-session cap on executed tool calls.
        rpm: LLM requests per minute.
        max_retries / retry_delay: Retry policy for the LLM call.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tool_calls: int = 12,
        rpm: int = 200,
        max_retries: int = 4,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
    ):
        provider = provider.lower()
        if provider not in ("openai", "azure", "anthropic", "mistral"):
            raise NotImplementedError(f"kg agent ingest supports openai, azure, anthropic, mistral; got {provider!r}")

        self.model = model
        self.provider = provider
        self.max_tool_calls = max_tool_calls
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.limiter = AsyncLimiter(rpm, 60)
        self.stats = IngestStats()

        if provider == "anthropic":
            import anthropic
            kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": 0}
            if api_key:
                kwargs["api_key"] = api_key
            self._client = anthropic.AsyncAnthropic(**kwargs)
        elif provider == "azure":
            import openai
            self._client = openai.AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                timeout=openai.Timeout(timeout, connect=10.0),
            )
        else:  # openai, mistral: OpenAI-compatible chat.completions with tools
            import openai
            compat = OPENAI_COMPATIBLE[provider]
            kwargs = {"timeout": openai.Timeout(timeout, connect=10.0), "max_retries": 0}
            key = api_key or os.getenv(compat["key_env"])
            if key:
                kwargs["api_key"] = key
            resolved_base = base_url or compat["base_url"]
            if resolved_base:
                kwargs["base_url"] = resolved_base
            self._client = openai.AsyncOpenAI(**kwargs)

    # -- Raw provider calls (patched in tests) ---------------------------------

    async def _with_retry(self, coro_factory: Callable[[], Awaitable[Any]]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    return await coro_factory()
            except Exception as exc:
                last_exc = exc
                logger.warning("agent LLM call %d/%d failed: %s", attempt + 1, self.max_retries, str(exc)[:200])
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        raise RuntimeError(f"agent LLM call failed after {self.max_retries} attempts: {last_exc}")

    async def _chat_openai(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        # `tools` is the contract's list object, built and sorted once, so the
        # serialized prefix is stable and OpenAI's automatic prefix cache applies.
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        if self.provider in ("openai", "azure") and _is_openai_reasoning_model(self.model):
            kwargs["reasoning_effort"] = "minimal"
        return await self._with_retry(lambda: self._client.chat.completions.create(**kwargs))

    async def _chat_anthropic(self, system: list[dict[str, Any]], messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,        # list of blocks; last carries cache_control
            "messages": messages,
            "tools": tools,          # last tool carries cache_control
            "max_tokens": ANTHROPIC_MAX_TOKENS,
        }
        # Thinking is off unless explicitly enabled: not passing `thinking` keeps it disabled.
        return await self._with_retry(lambda: self._client.messages.create(**kwargs))

    # -- Provider-specific step: returns tool calls ------------------------------

    @staticmethod
    def _int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        # OpenAI: prompt_tokens/completion_tokens (+ prompt_tokens_details.cached_tokens);
        # Anthropic: input_tokens/output_tokens (+ cache_read_input_tokens, cache_creation_input_tokens).
        # Mistral (OpenAI-compatible) reports no cache fields; they stay 0.
        self.stats.prompt_tokens += self._int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None))
        self.stats.completion_tokens += self._int(getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None))
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if cached is None:
            cached = getattr(usage, "cache_read_input_tokens", None)
        self.stats.cached_prompt_tokens += self._int(cached)
        self.stats.cache_write_tokens += self._int(getattr(usage, "cache_creation_input_tokens", None))

    async def _step_openai(self, convo: list[dict[str, Any]], contract: McpToolContract) -> list[_ToolCall]:
        resp = await self._chat_openai(convo, contract.openai_tools)
        self.stats.llm_calls += 1
        self._record_usage(getattr(resp, "usage", None))
        msg = resp.choices[0].message
        raw_calls = list(getattr(msg, "tool_calls", None) or [])
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if raw_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in raw_calls
            ]
        convo.append(entry)
        return [_ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in raw_calls]

    async def _step_anthropic(self, convo: list[dict[str, Any]], contract: McpToolContract) -> list[_ToolCall]:
        resp = await self._chat_anthropic(contract.anthropic_system, convo, contract.anthropic_tools)
        self.stats.llm_calls += 1
        self._record_usage(getattr(resp, "usage", None))
        blocks = list(getattr(resp, "content", None) or [])
        serialized: list[dict[str, Any]] = []
        calls: list[_ToolCall] = []
        for b in blocks:
            btype = getattr(b, "type", None)
            if btype == "text":
                serialized.append({"type": "text", "text": b.text})
            elif btype == "tool_use":
                serialized.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                calls.append(_ToolCall(b.id, b.name, json.dumps(b.input if isinstance(b.input, dict) else {})))
            # thinking blocks are not requested; anything else is dropped
        if not serialized:
            serialized.append({"type": "text", "text": ""})
        convo.append({"role": "assistant", "content": serialized})
        return calls

    # -- One dataset session ---------------------------------------------------

    async def run_session(
        self,
        messages: list[dict[str, Any]],
        *,
        contract: McpToolContract,
        execute_tool: ToolExecutor,
        date_str: str | None,
    ) -> None:
        """Let the agent process one dataset session. Tool results flow back to
        the model; what got created is tracked by the executor, not here."""
        speaker_a, speaker_b = infer_speakers(messages)
        header = USER_HEADER_TEMPLATE.format(
            date=date_str or "an unknown date", speaker_a=speaker_a, speaker_b=speaker_b,
        )
        user_text = f"{header}\n{render_session(messages)}"

        anthropic = self.provider == "anthropic"
        convo: list[dict[str, Any]] = (
            [{"role": "user", "content": user_text}]
            if anthropic
            else [{"role": "system", "content": contract.system_text}, {"role": "user", "content": user_text}]
        )

        self.stats.sessions += 1
        calls_this_session = 0
        hit_cap = False

        while True:
            try:
                tool_calls = await (self._step_anthropic(convo, contract) if anthropic
                                    else self._step_openai(convo, contract))
            except Exception as exc:
                self.stats.llm_failures += 1
                logger.error("agent session aborted: %s", str(exc)[:200])
                break

            self.stats.tool_calls_per_llm_call[len(tool_calls)] += 1
            if not tool_calls:
                break

            results: list[tuple[str, str]] = []
            for tc in tool_calls:
                if calls_this_session >= self.max_tool_calls:
                    hit_cap = True
                    results.append((tc.id, "Tool budget for this session exhausted; call skipped."))
                    continue
                try:
                    args = json.loads(tc.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    self.stats.tool_errors_by_name[tc.name] += 1
                    results.append((tc.id, f"Invalid tool arguments: {exc}"))
                    continue
                calls_this_session += 1
                self.stats.tool_calls_total += 1
                self.stats.tool_calls_by_name[tc.name] += 1
                try:
                    text = await execute_tool(tc.name, args)
                except Exception as exc:
                    self.stats.tool_errors_by_name[tc.name] += 1
                    text = f"Tool error: {str(exc)[:500]}"
                results.append((tc.id, text[:8000]))

            if anthropic:
                convo.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": cid, "content": text} for cid, text in results
                ]})
            else:
                for cid, text in results:
                    convo.append({"role": "tool", "tool_call_id": cid, "content": text})

            if hit_cap:
                self.stats.max_tool_calls_hit += 1
                logger.warning("max_tool_calls=%d hit for session dated %s; stopping loop", self.max_tool_calls, date_str)
                break

        self.stats.tool_calls_per_session.append(calls_this_session)


# ---------------------------------------------------------------------------
# Scripted stand-in (no LLM) for CI smoke
# ---------------------------------------------------------------------------


class ScriptedIngestor:
    """Fixed sequence of tool calls through the same executor/guardrails:
    session start, one kg_query, three kg_node_create calls (one with relates_to),
    one kg_node_update (status touch, which reactivates), session end. Exercises
    the plumbing without an LLM."""

    model = "stub"
    provider = "none"

    def __init__(self) -> None:
        self.stats = IngestStats()

    async def run_session(
        self,
        messages: list[dict[str, Any]],
        *,
        contract: McpToolContract,
        execute_tool: ToolExecutor,
        date_str: str | None,
    ) -> None:
        self.stats.sessions += 1
        speaker_a, _ = infer_speakers(messages)
        texts = [m.get("content", "") for m in messages if m.get("content")]
        created: list[str] = []
        n_calls = 0

        async def call(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
            nonlocal n_calls
            n_calls += 1
            self.stats.tool_calls_total += 1
            self.stats.tool_calls_by_name[name] += 1
            try:
                out = await execute_tool(name, args)
            except Exception as exc:
                self.stats.tool_errors_by_name[name] += 1
                logger.warning("stub agent: %s failed: %s", name, exc)
                return None
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                return None

        await call("kg_session_start", {})
        await call("kg_query", {"text": texts[0][:200] if texts else "context", "limit": 5})
        for i in range(3):
            content = f"{speaker_a} said: {texts[i % len(texts)][:300]}" if texts else f"placeholder {i}"
            args: dict[str, Any] = {"type": "observation", "content": content, "tags": ["stub"]}
            if i == 2 and created:
                args["relates_to"] = [created[0]]
            res = await call("kg_node_create", args)
            if res and res.get("id"):
                created.append(res["id"])
        if created:
            await call("kg_node_update", {"id": created[0], "status": "open"})
        await call("kg_session_end", {})
        self.stats.tool_calls_per_session.append(n_calls)
