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
every tool with ``currentSession = null`` and ``kg_node`` refuses to create
without an active session. The MCP route resolves the tenant's active session
per request, which is exactly the path production agents use.

Mapping
-------
- ``user_id``  -> one KG **project** named ``bench-{user_id}``. Project is the
  unit ``applySessionDecay`` scopes by, so it is the isolation unit here too.
  Every ``kg_node`` create and every ``kg_query`` passes ``project`` explicitly.
- one dataset session (a distinct ``timestamp`` for a user) -> one KG session
  (``kg_session start`` / ``end``), so session decay runs through the real
  path, in dataset order. ``decay="off"`` never ends/starts sessions between
  dataset sessions (one long-lived session per run), so decay never fires.
- the dataset session date is stored as a tag ``date:YYYY-MM-DD`` (never in
  ``content``, which would skew embeddings) and prepended to the memory text
  when search results are formatted for the answerer.
- ``delete_user`` wipes the project via a **root** SurrealDB connection on the
  local instance (there is no tenant-scoped wipe route). Refuses ``KG_DB_NAME=main``.

Concurrency
-----------
KG sessions are tenant-global (one active session per tenant), so ingest is
serialized with a lock and the runners force ``--max-workers 1`` for this
backend. Search is read-only (``read_only=true``) and runs concurrently.
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

from benchmarks.common.kg_extract import (
    PROMPT_VERSION,
    KgExtractor,
    prompt_sha256,
)

logger = logging.getLogger(__name__)

DEFAULT_KG_URL = "http://localhost:3000"
CREDENTIALS_PATH = Path.home() / ".kg" / "credentials.json"
MAX_QUERY_LIMIT = 100  # kg_query limit cap (contracts.ts)
MAX_RELATES_PER_CREATE = 20  # kg_node relates_to cap (contracts.ts)
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
        extractor_model / extractor_provider: LLM used to turn turns into nodes.
        rpm: Client-side KG requests per minute cap.
        llm_rpm: Extraction LLM requests per minute.
        max_retries / retry_delay / timeout: HTTP retry policy.
        db_url / db_user / db_pass / db_ns / db_name: root SurrealDB access for
            ``delete_user``. Env: KG_DB_URL (http form), KG_DB_USER, KG_DB_PASS,
            KG_DB_NS, KG_DB_NAME.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        spread: bool = True,
        decay: str = "on",
        extractor_model: str = "gpt-4o-mini",
        extractor_provider: str = "openai",
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
        self.extractor = KgExtractor(model=extractor_model, provider=extractor_provider, rpm=llm_rpm)

        self.db_url = (db_url or os.getenv("KG_DB_URL", "http://localhost:8000")).rstrip("/")
        self.db_user = db_user or os.getenv("KG_DB_USER", "root")
        self.db_pass = db_pass or os.getenv("KG_DB_PASS", "root")
        self.db_ns = db_ns or os.getenv("KG_DB_NS", "kg")
        self.db_name = db_name or os.getenv("KG_DB_NAME", "")

        self._session: aiohttp.ClientSession | None = None
        self._rpc_id = 0
        self._initialized = False

        # Ingest state. Sessions are tenant-global, hence one lock for all users.
        self._ingest_lock = asyncio.Lock()
        self._open_session: dict[str, Any] | None = None  # {"id", "user_id", "key"}
        self._projects_ready: set[str] = set()
        self._node_ids: dict[str, list[str]] = {}       # user_id -> created node ids (global index order)
        self._node_contents: dict[str, list[str]] = {}  # user_id -> contents (same order)
        self._last_session_key: dict[str, Any] = {}     # user_id -> last dataset-session key

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
            if self._open_session is not None:
                await self._end_session()
        except Exception as exc:
            logger.warning("Failed to end KG session on close: %s", exc)
        if self._session and not self._session.closed:
            await self._session.close()

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
                            raise KgAuthError(KG_TOKEN_HINT)
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
        await self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "memory-benchmarks-kg", "version": "0.1"}})
        self._initialized = True

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_initialized()
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise KgError(f"{name}: unexpected result {result!r}")
        content = result.get("content") or []
        text = content[0].get("text", "") if content and isinstance(content[0], dict) else ""
        if result.get("isError"):
            raise KgToolError(f"{name}: {text}")
        try:
            return json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise KgError(f"{name}: non-JSON tool result: {text[:200]}") from exc

    # ------------------------------------------------------------------
    # Session management (tenant-global; caller holds _ingest_lock)
    # ------------------------------------------------------------------

    @staticmethod
    def project_for(user_id: str) -> str:
        name = f"bench-{user_id}"
        if not PROJECT_NAME_RE.match(name):
            raise KgError(f"user_id {user_id!r} produces an unsafe project name")
        return name

    async def _start_session(self, user_id: str, key: Any) -> dict[str, Any]:
        project = self.project_for(user_id)
        resp = await self._call_tool("kg_session", {
            "action": "start",
            "project": project,
            "create_project": True,
            "notes": f"memory-benchmarks {user_id} session={key}",
        })
        self._projects_ready.add(project)
        self._open_session = {"id": resp.get("session_id"), "user_id": user_id, "key": key}
        logger.debug("KG session started for %s key=%s decayed=%s", user_id, key, resp.get("nodes_decayed"))
        return resp

    async def _end_session(self) -> None:
        if self._open_session is None:
            return
        try:
            await self._call_tool("kg_session", {"action": "end"})
        finally:
            self._open_session = None

    async def _ensure_session_for_add(self, user_id: str, key: Any) -> None:
        """Bring the tenant's active session into the state this add needs."""
        project = self.project_for(user_id)
        if self.decay == "off":
            # Only ever start a session when a project must be created (an
            # empty project decays nothing), and never end one mid-run.
            if project not in self._projects_ready:
                await self._start_session(user_id, key)
            elif self._open_session is None:
                await self._start_session(user_id, key)
            return

        # decay == "on": one KG session per (user_id, dataset session).
        cur = self._open_session
        if cur is not None and cur["user_id"] == user_id and cur["key"] == key:
            return
        if cur is not None:
            await self._end_session()
        await self._start_session(user_id, key)

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
        """Extract nodes from ``messages`` and record them. Returns
        ``{"results": [{"id", "memory", "event": "ADD"}, ...]}`` or None on failure."""
        date_tag = _date_tag_from(timestamp, observation_date)
        session_key = timestamp if timestamp is not None else (observation_date or "undated")
        project = self.project_for(user_id)

        ids = self._node_ids.setdefault(user_id, [])
        contents = self._node_contents.setdefault(user_id, [])

        async with self._ingest_lock:
            try:
                await self._ensure_session_for_add(user_id, session_key)
                self._last_session_key[user_id] = session_key

                prior = list(enumerate(contents))
                extracted = await self.extractor.extract(messages, prior=prior, date_str=date_tag)

                results: list[dict[str, Any]] = []
                base_index = len(ids)
                for node in extracted:
                    relates_to = [ids[i] for i in node["relates_to_indices"] if 0 <= i < base_index][:MAX_RELATES_PER_CREATE]
                    tags = list(node["tags"])
                    if date_tag:
                        tags.append(f"date:{date_tag}")
                    tags.append(f"bench:{user_id}")
                    args: dict[str, Any] = {
                        "action": "create",
                        "type": node["type"],
                        "content": node["content"],
                        "project": project,
                        "tags": tags,
                    }
                    if relates_to:
                        args["relates_to"] = relates_to
                    created = await self._call_tool("kg_node", args)
                    node_id = created.get("id", "")
                    ids.append(node_id)
                    contents.append(node["content"])
                    results.append({"id": node_id, "memory": node["content"], "event": "ADD",
                                    "edges_created": created.get("edges_created", 0)})
                return {"results": results}
            except KgAuthError:
                raise
            except Exception as exc:
                logger.error("KG add failed for user=%s: %s", user_id, str(exc)[:300])
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

        Returns ``{"results": [...], "query_debug": {...}}`` — ``format_search_results``
        accepts this dict form and surfaces ``query_debug`` into the result JSON,
        which is where the pre/post-truncation counts live (spec CAP-6).
        """
        if self._open_session is not None:
            # All ingest for this run is done by the time search starts; close
            # the trailing session so the last dataset session is "ended" like
            # every other one. Harmless if another search already did it.
            async with self._ingest_lock:
                await self._end_session()

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
    # Delete (root SurrealDB, local instance only)
    # ------------------------------------------------------------------

    async def delete_user(self, user_id: str) -> bool:
        project = self.project_for(user_id)
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
        self._node_ids.pop(user_id, None)
        self._node_contents.pop(user_id, None)
        self._last_session_key.pop(user_id, None)
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


def build_kg_metadata(args: Any) -> dict[str, Any]:
    """Metadata fields for a ``--backend kg`` run (spec CAP-7)."""
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
        "kg_extract_model": getattr(args, "kg_extract_model", "gpt-4o-mini"),
        "kg_extract_prompt_version": PROMPT_VERSION,
        "kg_extract_prompt_sha256": prompt_sha256(),
        "kg_embed_provider": os.getenv("KG_EMBED_PROVIDER"),
    }
