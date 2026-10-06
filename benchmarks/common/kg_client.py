"""
Komplyt KG Client
=================

Async client that lets the unmodified benchmark runners drive the Komplyt
Knowledge Graph (https://github.com/Komplyzen/Knowledge-Graph) as the memory
system under test. Same public surface as ``Mem0Client``:

  client.add(messages, user_id, timestamp=...)
  client.search(query, user_id, top_k=...)
  client.delete_user(user_id)

Transport
---------
MCP JSON-RPC over HTTP: ``POST {KG_URL}/mcp`` with a bearer token, method
``tools/call``. The REST routes under ``/api/v1`` are NOT used: they execute
every tool with ``currentSession = null`` and ``kg_node_create`` refuses to create
without an active session. The MCP route resolves the tenant's active session
per request, which is exactly the path production agents use.

Ingest: "Komplyt Zero as shipped"
---------------------------------
``add`` hands each dataset session to an LLM agent equipped with the KG's own
MCP tools (``kg_agent_ingest.AgentIngestor``). The model decides what to call
(kg_session_start, kg_query, kg_node_create, kg_node_update, ...). The harness enforces only:

- every ``kg_node_create`` gets ``project=bench-{user_id}`` and
  a ``date:YYYY-MM-DD`` tag (the date never goes into ``content``);
- a write without an active session first gets a harness-started session;
- ``kg_session_start`` by the model is deduplicated against the session the
  harness already opened for this (user, dataset session), and forced onto the
  benchmark project, so decay fires once per dataset session, not per whim;
- if the model did not end the session, the harness ends it after the loop.
- ``decay="off"``: one long-lived session per run; model start/end calls are
  acknowledged but not executed, so decay never fires.

Mapping
-------
- ``user_id`` -> one KG **project** ``bench-{user_id}``, the unit
  ``applySessionDecay`` scopes by. Every query passes ``project`` explicitly.
- ``search`` is one ``kg_query`` with ``read_only=true`` (no implicit
  activation), spread per variant, truncated to ``top_k``.
- ``delete_user`` wipes the project via a **root** SurrealDB connection on the
  local instance (no tenant-scoped wipe route exists). Refuses ``KG_DB_NAME=main``.

Concurrency
-----------
KG sessions are tenant-global, so ingest is serialized with a lock and the
runners force ``--max-workers 1`` for this backend. Search runs concurrently.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from aiolimiter import AsyncLimiter

from benchmarks.common.kg_agent_ingest import (
    AgentIngestor,
    McpToolContract,
    ScriptedIngestor,
    contract_from_mcp,
)

logger = logging.getLogger(__name__)

DEFAULT_KG_URL = "http://localhost:3000"
CREDENTIALS_PATH = Path.home() / ".kg" / "credentials.json"
MAX_QUERY_LIMIT = 100  # kg_query limit cap (contracts.ts)
SPREAD_HOPS = 2
# Client-side cap, well under the KG's per-principal MCP limit (1000/min) and
# the REST limit (500/min), so the server-side limiter is never the thing
# pacing ingest (spec CAP-8).
DEFAULT_RPM = 450
PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
KG_TOKEN_HINT = (
    "KG returned 401 Unauthorized. Run `kg auth login` (Knowledge-Graph/src/bin/kg.ts) "
    "against staging Komplyt, or set KG_TOKEN."
)
# spec-benchmark-ci CAP-3: in CI the token is signed in-job against a local JWKS
# server, so a 401 means "token or JWKS server", and one retry covers a JWKS
# that was not yet served when the KG server first fetched it.
KG_CI_TOKEN_HINT = "KG returned 401 Unauthorized: token or JWKS server (KG_CI mode). Check the in-job signer and the JWKS static server."
KG_CI = bool(os.getenv("KG_CI"))
WRITE_TOOLS = frozenset({"kg_node_create", "kg_node_update", "kg_relate"})


class KgError(RuntimeError):
    pass


class KgAuthError(KgError):
    pass


class KgToolError(KgError):
    """The tool ran and returned isError=true (e.g. NO_SESSION, NOT_FOUND)."""


def load_token(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.getenv("KG_TOKEN")
    if env:
        return env
    try:
        creds = json.loads(CREDENTIALS_PATH.read_text())
        token = creds.get("access_token")
        if token:
            exp = creds.get("access_token_expires_at")
            if exp:
                try:
                    if datetime.fromisoformat(exp.replace("Z", "+00:00")) < datetime.now(timezone.utc):
                        logger.warning("Token in %s is expired; run `kg auth login`", CREDENTIALS_PATH)
                except ValueError:
                    pass
            return token
    except (OSError, json.JSONDecodeError):
        pass
    raise KgAuthError(f"No KG token: set KG_TOKEN or run `kg auth login` (looked in {CREDENTIALS_PATH})")


def _date_tag_from(timestamp: int | None, observation_date: str | None) -> str | None:
    if timestamp is not None:
        try:
            return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return None
    if observation_date:
        return observation_date[:10]
    return None


def _date_from_tags(tags: list[str] | None) -> str | None:
    for t in tags or []:
        if isinstance(t, str) and t.startswith("date:"):
            return t[5:]
    return None


class KgClient:
    """Async Komplyt KG client with the Mem0Client surface.

    Args:
        url: KG HTTP server base URL (``/mcp`` is appended). Env: KG_URL.
        token: Bearer token. Falls back to KG_TOKEN, then ~/.kg/credentials.json.
        spread: Send ``spread=true`` on kg_query (spreading activation).
        decay: "on" = one KG session per dataset session (decay fires between
            them); "off" = one long-lived session, decay never fires.
        agent_model / agent_provider: LLM driving the MCP tools during ingest.
        max_tool_calls: Per-dataset-session cap on agent tool calls.
        ingestor: Override the ingestor (e.g. ``ScriptedIngestor()`` for smoke).
        rpm: Client-side KG requests per minute cap.
        llm_rpm: Agent LLM requests per minute.
        max_retries / retry_delay / timeout: HTTP retry policy.
        db_*: root SurrealDB access for ``delete_user``. Env: KG_DB_URL (http
            form), KG_DB_USER, KG_DB_PASS, KG_DB_NS, KG_DB_NAME.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        spread: bool = True,
        decay: str = "on",
        agent_model: str = "gpt-5-mini",
        agent_provider: str = "openai",
        max_tool_calls: int = 12,
        ingestor: Any | None = None,
        rpm: int = DEFAULT_RPM,
        llm_rpm: int = 200,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
        db_url: str | None = None,
        db_user: str | None = None,
        db_pass: str | None = None,
        db_ns: str | None = None,
        db_name: str | None = None,
    ):
        if decay not in ("on", "off"):
            raise ValueError("decay must be 'on' or 'off'")
        self.url = (url or os.getenv("KG_URL", DEFAULT_KG_URL)).rstrip("/")
        self.token = load_token(token)
        self.spread = spread
        self.decay = decay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.limiter = AsyncLimiter(min(rpm, DEFAULT_RPM), 60)
        self.ingestor = ingestor or AgentIngestor(
            model=agent_model, provider=agent_provider, max_tool_calls=max_tool_calls, rpm=llm_rpm,
        )
        self.ingest_mode = "scripted-stub" if isinstance(self.ingestor, ScriptedIngestor) else "mcp-agent"

        self.db_url = (db_url or os.getenv("KG_DB_URL", "http://localhost:8000")).rstrip("/")
        self.db_user = db_user or os.getenv("KG_DB_USER", "root")
        self.db_pass = db_pass or os.getenv("KG_DB_PASS", "root")
        self.db_ns = db_ns or os.getenv("KG_DB_NS", "kg")
        self.db_name = db_name or os.getenv("KG_DB_NAME", "")

        self._session: aiohttp.ClientSession | None = None
        self._rpc_id = 0
        self._initialized = False
        self._init_result: dict[str, Any] | None = None
        self._contract: McpToolContract | None = None

        # Ingest state. Sessions are tenant-global, hence one lock for all users.
        self._ingest_lock = asyncio.Lock()
        # The runners call add() per chunk (LoCoMo: one turn; LongMemEval: one
        # user/assistant pair). The agent must see a whole dataset session, so
        # chunks are buffered until the (user_id, timestamp) key changes and
        # flushed as one agent session (also on search/close/delete_user).
        self._pending: dict[str, Any] | None = None  # {"user_id", "key", "date_tag", "messages"}
        self._open_session: dict[str, Any] | None = None  # {"id", "user_id", "key"}
        self._projects_ready: set[str] = set()
        self._created: dict[str, list[tuple[str, str]]] = {}  # user_id -> [(node_id, content)]
        self.guardrails: dict[str, int] = {
            "session_started_by_harness": 0,
            "session_ended_by_harness": 0,
            "session_start_deduped": 0,
            "session_end_deferred": 0,
            "project_injected": 0,
            "date_tag_added": 0,
            "tool_errors_returned_to_model": 0,
        }

    # ------------------------------------------------------------------
    # HTTP / JSON-RPC plumbing
    # ------------------------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=self.timeout,
                connector=aiohttp.TCPConnector(limit=16),
            )
        return self._session

    async def close(self) -> None:
        try:
            async with self._ingest_lock:
                await self._flush_pending()
                if self._open_session is not None:
                    await self._end_session(by_harness=True)
        except Exception as exc:
            logger.warning("Failed to flush/end KG session on close: %s", exc)
        if self._session and not self._session.closed:
            await self._session.close()
        stats = getattr(self.ingestor, "stats", None)
        if stats is not None and stats.sessions:
            line = stats.summary_line()
            logger.info(line)
            print(f"  {line}")

    async def __aenter__(self) -> KgClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """One JSON-RPC request with retry on 429/5xx/transport errors."""
        session = await self._get_session()
        self._rpc_id += 1
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method}
        if params is not None:
            body["params"] = params

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(f"{self.url}/mcp", json=body) as resp:
                        if resp.status == 401:
                            if KG_CI and attempt == 0:
                                logger.warning("KG 401 in KG_CI mode; retrying once (token or JWKS server)")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            raise KgAuthError(KG_CI_TOKEN_HINT if KG_CI else KG_TOKEN_HINT)
                        if resp.status == 403:
                            raise KgAuthError(f"KG returned 403: {await resp.text()}")
                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", "0") or 0)
                            delay = max(retry_after, self.retry_delay * (2 ** attempt))
                            logger.warning("KG 429 (remaining=%s); sleeping %.1fs", resp.headers.get("X-RateLimit-Remaining"), delay)
                            await asyncio.sleep(delay)
                            continue
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp.status, message=await resp.text())
                        if resp.status >= 400:
                            raise KgError(f"KG HTTP {resp.status}: {(await resp.text())[:300]}")
                        data = await resp.json()
            except KgAuthError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, KgError) as exc:
                last_exc = exc
                logger.warning("KG %s attempt %d/%d failed: %s", method, attempt + 1, self.max_retries, str(exc)[:200])
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                continue

            if "error" in data:
                err = data["error"]
                raise KgError(f"JSON-RPC error {err.get('code')}: {err.get('message')}")
            return data.get("result")

        raise KgError(f"KG {method} failed after {self.max_retries} attempts: {last_exc}")

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        result = await self._rpc("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "memory-benchmarks-kg", "version": "0.2"},
        })
        self._init_result = result if isinstance(result, dict) else {}
        self._initialized = True

    async def _call_tool_raw(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any] | None, str, bool]:
        """Returns (parsed_json_or_None, text, is_error)."""
        await self._ensure_initialized()
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise KgError(f"{name}: unexpected result {result!r}")
        content = result.get("content") or []
        text = content[0].get("text", "") if content and isinstance(content[0], dict) else ""
        is_error = bool(result.get("isError"))
        parsed: dict[str, Any] | None = None
        if not is_error and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        return parsed, text, is_error

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        parsed, text, is_error = await self._call_tool_raw(name, arguments)
        if is_error:
            raise KgToolError(f"{name}: {text}")
        if parsed is None:
            if not text:
                return {}
            raise KgError(f"{name}: non-JSON tool result: {text[:200]}")
        return parsed

    # ------------------------------------------------------------------
    # Tool contract (fetched once)
    # ------------------------------------------------------------------

    async def tool_contract(self) -> McpToolContract:
        if self._contract is None:
            await self._ensure_initialized()
            listed = await self._rpc("tools/list", {})
            tools = listed.get("tools", []) if isinstance(listed, dict) else []
            instructions = (self._init_result or {}).get("instructions")
            self._contract = contract_from_mcp(tools, instructions if isinstance(instructions, str) else None)
            logger.info("KG tool contract: %d tools, sha256=%s", len(tools), self._contract.sha256[:12])
        return self._contract

    @property
    def tool_contract_sha256(self) -> str | None:
        return self._contract.sha256 if self._contract else None

    # ------------------------------------------------------------------
    # Session management (tenant-global; caller holds _ingest_lock)
    # ------------------------------------------------------------------

    @staticmethod
    def project_for(user_id: str) -> str:
        name = f"bench-{user_id}"
        if not PROJECT_NAME_RE.match(name):
            raise KgError(f"user_id {user_id!r} produces an unsafe project name")
        return name

    def _session_matches(self, user_id: str, key: Any) -> bool:
        cur = self._open_session
        return cur is not None and cur["user_id"] == user_id and cur["key"] == key

    async def _start_session(self, user_id: str, key: Any, notes: str | None = None) -> dict[str, Any]:
        project = self.project_for(user_id)
        resp = await self._call_tool("kg_session_start", {
            "project": project,
            "create_project": True,
            "notes": notes or f"memory-benchmarks {user_id} session={key}",
        })
        self._projects_ready.add(project)
        self._open_session = {"id": resp.get("session_id"), "user_id": user_id, "key": key}
        logger.debug("KG session started for %s key=%s decayed=%s", user_id, key, resp.get("nodes_decayed"))
        return resp

    async def _end_session(self, by_harness: bool = False) -> dict[str, Any]:
        if self._open_session is None:
            return {}
        try:
            resp = await self._call_tool("kg_session_end", {})
        finally:
            self._open_session = None
        if by_harness:
            self.guardrails["session_ended_by_harness"] += 1
        return resp

    async def _ensure_session_for_write(self, user_id: str, key: Any) -> None:
        """Harness-side: make sure a session that fits this add is active."""
        project = self.project_for(user_id)
        if self.decay == "off":
            # Start only when the project must be created (an empty project
            # decays nothing) or no session is open at all; never end mid-run.
            if project not in self._projects_ready or self._open_session is None:
                await self._start_session(user_id, key)
                self.guardrails["session_started_by_harness"] += 1
            return
        if self._session_matches(user_id, key):
            return
        if self._open_session is not None:
            await self._end_session(by_harness=True)
        await self._start_session(user_id, key)
        self.guardrails["session_started_by_harness"] += 1

    # ------------------------------------------------------------------
    # Guardrailed tool execution for the agent
    # ------------------------------------------------------------------

    def _make_executor(self, user_id: str, key: Any, date_tag: str | None):
        project = self.project_for(user_id)
        created = self._created.setdefault(user_id, [])

        async def execute(name: str, args: dict[str, Any]) -> str:
            args = dict(args)
            try:
                if name == "kg_session_start":
                    return await self._exec_session_start(args, user_id, key)
                if name == "kg_session_end":
                    return await self._exec_session_end()

                if name in ("kg_node_create", "kg_node_update"):
                    is_create = name == "kg_node_create"
                    await self._ensure_session_for_write(user_id, key)
                    if is_create:
                        if args.get("project") != project:
                            self.guardrails["project_injected"] += 1
                        args["project"] = project
                        args.pop("container", None)
                        tags = [str(t) for t in (args.get("tags") or []) if str(t).strip()]
                        if date_tag and f"date:{date_tag}" not in tags:
                            tags.append(f"date:{date_tag}")
                            self.guardrails["date_tag_added"] += 1
                        tags.append(f"bench:{user_id}")
                        args["tags"] = tags
                    parsed, text, is_error = await self._call_tool_raw(name, args)
                    if is_error:
                        self.guardrails["tool_errors_returned_to_model"] += 1
                        return f"Error: {text}"
                    if is_create and parsed and parsed.get("id"):
                        created.append((parsed["id"], parsed.get("content", args.get("content", ""))))
                    return text

                if name in WRITE_TOOLS:  # kg_relate
                    await self._ensure_session_for_write(user_id, key)
                elif name == "kg_query":
                    if args.get("project") != project:
                        self.guardrails["project_injected"] += 1
                    args["project"] = project

                _, text, is_error = await self._call_tool_raw(name, args)
                if is_error:
                    self.guardrails["tool_errors_returned_to_model"] += 1
                    return f"Error: {text}"
                return text
            except KgAuthError:
                raise
            except KgError as exc:
                self.guardrails["tool_errors_returned_to_model"] += 1
                return f"Error: {exc}"

        return execute

    async def _exec_session_start(self, args: dict[str, Any], user_id: str, key: Any) -> str:
        if self._session_matches(user_id, key) or (self.decay == "off" and self._open_session is not None
                                                    and self.project_for(user_id) in self._projects_ready):
            self.guardrails["session_start_deduped"] += 1
            return json.dumps({
                "action": "started",
                "session_id": self._open_session["id"],
                "project": self.project_for(user_id),
                "nodes_decayed": 0,
                "note": "Session already active for this conversation.",
            })
        if self._open_session is not None:
            if self.decay == "off":
                # Different project, decay off: the new project must exist; an
                # empty project decays nothing, so this start is harmless.
                pass
            else:
                await self._end_session(by_harness=True)
        resp = await self._start_session(user_id, key, notes=args.get("notes"))
        return json.dumps(resp)

    async def _exec_session_end(self) -> str:
        if self.decay == "off":
            self.guardrails["session_end_deferred"] += 1
            return json.dumps({"action": "ended", "session_id": (self._open_session or {}).get("id"),
                               "note": "Acknowledged."})
        if self._open_session is None:
            return "Error: No active session. Start a session first with kg_session_start."
        resp = await self._end_session()
        return json.dumps(resp)

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Buffer one chunk of a dataset session. When the (user_id, timestamp)
        key changes, the buffered session is handed to the agent as a whole.
        Returns ``{"results": [{"id", "memory", "event": "ADD"}, ...]}`` for
        nodes created by a flush triggered by this call (empty while buffering),
        or None if that flush failed."""
        date_tag = _date_tag_from(timestamp, observation_date)
        session_key = timestamp if timestamp is not None else (observation_date or "undated")
        self.project_for(user_id)  # validate early

        async with self._ingest_lock:
            flushed: dict | None = {"results": []}
            pending = self._pending
            if pending is not None and (pending["user_id"] != user_id or pending["key"] != session_key):
                flushed = await self._flush_pending()
            if self._pending is None:
                self._pending = {"user_id": user_id, "key": session_key, "date_tag": date_tag, "messages": []}
            self._pending["messages"].extend(messages)
            return flushed

    async def _flush_pending(self) -> dict | None:
        """Run the agent over the buffered dataset session. Caller holds the lock."""
        pending, self._pending = self._pending, None
        if pending is None or not pending["messages"]:
            return {"results": []}
        user_id, session_key, date_tag = pending["user_id"], pending["key"], pending["date_tag"]
        try:
            contract = await self.tool_contract()
            if self.decay == "on" and self._open_session is not None and not self._session_matches(user_id, session_key):
                # Previous dataset session left open by the model: close it so
                # decay fires exactly once per dataset session.
                await self._end_session(by_harness=True)

            created = self._created.setdefault(user_id, [])
            before = len(created)
            execute = self._make_executor(user_id, session_key, date_tag)
            await self.ingestor.run_session(pending["messages"], contract=contract, execute_tool=execute, date_str=date_tag)

            new = created[before:]
            return {"results": [{"id": nid, "memory": content, "event": "ADD"} for nid, content in new]}
        except KgAuthError:
            raise
        except Exception as exc:
            logger.error("KG ingest failed for user=%s session=%s: %s", user_id, session_key, str(exc)[:300])
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> dict[str, Any] | list[dict]:
        """Read-only kg_query scoped to the user's project.

        Returns ``{"results": [...], "query_debug": {...}}``; ``format_search_results``
        accepts this dict form and surfaces ``query_debug`` into the result JSON,
        which is where the pre/post-truncation counts live (spec CAP-6).
        """
        if self._pending is not None or self._open_session is not None:
            # Ingest for this conversation is done by the time search starts:
            # flush the buffered last dataset session and close the trailing KG
            # session so it is "ended" like every other one.
            async with self._ingest_lock:
                await self._flush_pending()
                if self._open_session is not None:
                    await self._end_session(by_harness=True)

        limit = max(1, min(int(top_k), MAX_QUERY_LIMIT))
        args: dict[str, Any] = {
            "text": query,
            "project": self.project_for(user_id),
            "read_only": True,
            "spread": self.spread,
            "spread_hops": SPREAD_HOPS,
            "limit": limit,
            "include_edges": False,
            "include_neighbors": False,
        }

        try:
            resp = await self._call_tool("kg_query", args)
        except KgAuthError:
            raise
        except Exception as exc:
            logger.error("KG search failed for user=%s: %s", user_id, str(exc)[:300])
            return []

        nodes = resp.get("nodes") or []
        ranked = self._rank(nodes)
        returned = len(ranked)
        ranked = ranked[:top_k]

        results = []
        for rank, n in enumerate(ranked):
            date = _date_from_tags(n.get("tags"))
            memory = f"[{date}] {n.get('content', '')}" if date else n.get("content", "")
            # Monotone score so format_search_results' re-sort keeps our order.
            score = float(returned - rank)
            entry: dict[str, Any] = {"memory": memory, "score": score, "id": n.get("id", "")}
            if n.get("created_at"):
                entry["created_at"] = n["created_at"]
            entry["score_debug"] = {
                "combined_score": n.get("combined_score"),
                "semantic_score": n.get("semantic_score"),
                "decay_score": n.get("decay_score"),
                "activation": n.get("activation"),
                "activation_source": n.get("activation_source", "direct"),
                "node_type": n.get("type"),
            }
            results.append(entry)

        query_debug = {
            "backend": "kg",
            "mode": resp.get("mode", "semantic"),
            "spread": self.spread,
            "hops": resp.get("hops"),
            "kg_limit": limit,
            "kg_returned": returned,
            "kg_after_truncation": len(results),
            "timeout": bool(resp.get("timeout", False)),
        }
        return {"results": results, "query_debug": query_debug}

    @staticmethod
    def _rank(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Seeds (direct hits) by combined_score desc, then spread-only nodes by
        activation desc. With spread=false every node is a seed."""
        direct = [n for n in nodes if n.get("activation_source", "direct") == "direct"]
        spread = [n for n in nodes if n.get("activation_source") == "spread"]
        direct.sort(key=lambda n: (n.get("combined_score") if n.get("combined_score") is not None else n.get("activation", 0.0)) or 0.0, reverse=True)
        spread.sort(key=lambda n: n.get("activation", 0.0) or 0.0, reverse=True)
        return direct + spread

    async def get_user_profile(self, user_id: str) -> dict | None:
        """Mem0 cloud feature; the KG has no equivalent."""
        return None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def ingest_metadata(self) -> dict[str, Any]:
        """Run-level ingest facts, merged into the result file's metadata."""
        meta: dict[str, Any] = {
            "kg_ingest_mode": self.ingest_mode,
            "kg_agent_model": getattr(self.ingestor, "model", None),
            "kg_agent_provider": getattr(self.ingestor, "provider", None),
            "kg_agent_max_tool_calls": getattr(self.ingestor, "max_tool_calls", None),
            "tool_contract_sha256": self.tool_contract_sha256,
            "tool_descriptions_truncated_for_openai": list(self._contract.truncated_descriptions) if self._contract else [],
            "kg_guardrails": dict(self.guardrails),
            "kg_nodes_created": sum(len(v) for v in self._created.values()),
        }
        stats = getattr(self.ingestor, "stats", None)
        if stats is not None:
            meta.update(stats.as_metadata())
        return meta

    # ------------------------------------------------------------------
    # Delete (root SurrealDB, local instance only)
    # ------------------------------------------------------------------

    async def delete_user(self, user_id: str) -> bool:
        project = self.project_for(user_id)
        if self._pending is not None and self._pending["user_id"] == user_id:
            self._pending = None  # never ingested; nothing to keep
        if not self.db_name:
            logger.error("delete_user: KG_DB_NAME not set; refusing to guess")
            return False
        if self.db_name == "main":
            logger.error("delete_user: refusing to wipe KG_DB_NAME=main")
            return False

        sql = f"""
LET $p = (SELECT VALUE id FROM project WHERE name = '{project}')[0];
DELETE decay_event WHERE node.project = $p;
DELETE edge WHERE in.project = $p OR out.project = $p;
DELETE node WHERE project = $p;
DELETE session WHERE project = $p;
DELETE project WHERE id = $p;
"""
        auth = base64.b64encode(f"{self.db_user}:{self.db_pass}".encode()).decode()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Basic {auth}",
            "surreal-ns": self.db_ns,
            "surreal-db": self.db_name,
        }
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as s:
                async with s.post(f"{self.db_url}/sql", data=sql, headers=headers) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        logger.warning("delete_user %s: SurrealDB HTTP %d: %s", user_id, resp.status, body[:300])
                        return False
                    try:
                        statements = json.loads(body)
                    except json.JSONDecodeError:
                        statements = []
                    errs = [st for st in statements if isinstance(st, dict) and st.get("status") == "ERR"]
                    if errs:
                        logger.warning("delete_user %s: %s", user_id, errs[0].get("result"))
                        return False
        except Exception as exc:
            logger.warning("delete_user %s failed: %s", user_id, exc)
            return False

        if self._open_session and self._open_session.get("user_id") == user_id:
            self._open_session = None
        self._projects_ready.discard(project)
        self._created.pop(user_id, None)
        logger.info("Wiped KG project %s", project)
        return True


# ---------------------------------------------------------------------------
# Result metadata helper (spliced into the runners' metadata dicts)
# ---------------------------------------------------------------------------


def _kg_commit() -> str | None:
    repo = os.getenv("KG_REPO")
    if not repo:
        return None
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except Exception:
        return None


def kg_variant(spread: bool, decay: str) -> str:
    if spread and decay == "on":
        return "full"
    if not spread and decay == "on":
        return "no-spread"
    if spread and decay == "off":
        return "no-decay"
    return "no-spread-no-decay"


def build_backend_metadata(args: Any, backend: str) -> dict[str, Any]:
    """Static metadata for non-mem0 backends (spec CAP-7). Run-level counters
    are added later via ``client.ingest_metadata()``."""
    if backend == "kg":
        spread = getattr(args, "kg_spread", "on") == "on"
        decay = getattr(args, "kg_decay", "on")
        return {
            "memory_backend": "kg",
            "kg_url": getattr(args, "kg_url", None) or os.getenv("KG_URL", DEFAULT_KG_URL),
            "kg_variant": kg_variant(spread, decay),
            "kg_spread": spread,
            "kg_decay": decay,
            "kg_spread_hops": SPREAD_HOPS,
            "kg_query_limit_cap": MAX_QUERY_LIMIT,
            "kg_commit": _kg_commit(),
            "kg_ingest_mode": "mcp-agent",
            "kg_agent_model": getattr(args, "kg_agent_model", "gpt-5-mini"),
            "kg_agent_provider": getattr(args, "kg_agent_provider", "openai"),
            "kg_embed_provider": os.getenv("KG_EMBED_PROVIDER"),
        }
    if backend == "none":
        return {
            "memory_backend": "none",
            "none_context_token_budget": getattr(args, "none_context_tokens", None),
        }
    return {}


# Backwards-compatible alias used by earlier runner wiring.
def build_kg_metadata(args: Any) -> dict[str, Any]:
    return build_backend_metadata(args, "kg")
