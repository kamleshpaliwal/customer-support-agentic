"""One long-lived MCP client connection shared by all tool calls in this process.

Starting the MCP server for every call costs about 0.6-0.8 s (interpreter start, imports, DB
pool), far more than the 2 ms query it runs. The session keeps a single stdio server process
and multiplexes calls over it. It:

* connects lazily on first use (``start()`` pre-warms it at application startup);
* reconnects and retries once if the server process dies (all our tools are read-only);
* routes MCP elicitation prompts to the request that triggered them, using a context
  variable that the MCP SDK preserves because it runs the callback in the calling task;
* can be switched off (``MCP_PERSISTENT_SESSION=false``) to fall back to one process per call.

The SDK requires a connection to be opened and closed by the same task, so a dedicated owner
task holds it open until the session is closed or found broken.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

import anyio
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, CallToolResult, ElicitRequestFormParams, ElicitResult

from support_agents.mcp.elicitation import get_elicitation_broker
from support_agents.settings import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

# AWS_* carries the role credentials a container platform provides (for example
# AWS_CONTAINER_CREDENTIALS_RELATIVE_URI); the server needs them to call Bedrock.
# APP_PROFILE selects the same .env.<profile> file in the server process.
_FORWARDED_ENV_PREFIXES = ("ACADEMIC_", "OLLAMA_", "EMBEDDING_", "RERANK_", "AWS_", "APP_")

# Which browser request an elicitation prompt belongs to; set by the caller of call_tool.
_ELICITATION_REQUEST_ID: ContextVar[str | None] = ContextVar("elicitation_request_id", default=None)


def is_connection_lost(error: BaseException) -> bool:
    """True when the MCP server process or its pipe is gone, as opposed to a tool error."""
    if isinstance(error, MCPError):
        return error.code == CONNECTION_CLOSED
    return isinstance(
        error,
        anyio.ClosedResourceError
        | anyio.BrokenResourceError
        | anyio.EndOfStream
        | BrokenPipeError
        | ConnectionResetError,
    )


async def _on_elicitation(context: Any, params: Any) -> ElicitResult:
    del context
    request_id = _ELICITATION_REQUEST_ID.get()
    if request_id is None or not isinstance(params, ElicitRequestFormParams):
        return ElicitResult(action="decline")

    properties = params.requested_schema.get("properties", {})
    field_name, field_schema = next(iter(properties.items()), (None, {}))
    options = field_schema.get("enum", []) if isinstance(field_schema, dict) else []
    if not field_name or not options:
        return ElicitResult(action="decline")

    return await get_elicitation_broker().ask(
        request_id,
        message=params.message,
        field_name=field_name,
        options=[str(option) for option in options],
    )


def _server_parameters() -> StdioServerParameters:
    settings = get_settings()
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "support_agents.mcp.server"],
        cwd=str(PROJECT_ROOT),
        env={
            # Retrieval and embedding settings must reach the server even without a .env file.
            **{
                key: value
                for key, value in os.environ.items()
                if key.startswith(_FORWARDED_ENV_PREFIXES)
            },
            "DATABASE_URL": settings.database_url,
            "MCP_TRANSPORT": "stdio",
        },
    )


class McpSession:
    def __init__(self, *, persistent: bool = True, connect_timeout_seconds: float = 30.0) -> None:
        self.persistent = persistent
        self._connect_timeout = connect_timeout_seconds
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self._client: Client | None = None
        self._owner: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._generation = 0

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> CallToolResult:
        """Call a tool. ``request_id`` routes any elicitation prompt to that browser request."""
        token = _ELICITATION_REQUEST_ID.set(request_id)
        try:
            if not self.persistent:
                async with Client(_server_parameters(), elicitation_callback=_on_elicitation) as c:
                    return await c.call_tool(name, arguments)
            for attempt in (1, 2):
                client, generation = await self._ensure_connected()
                try:
                    return await client.call_tool(name, arguments)
                except Exception as error:
                    if attempt == 2 or not is_connection_lost(error):
                        raise
                    logger.warning("MCP server connection lost; reconnecting and retrying once")
                    await self._discard(generation)
            raise AssertionError("unreachable")
        finally:
            _ELICITATION_REQUEST_ID.reset(token)

    async def start(self) -> None:
        """Pre-warm the connection so the first request does not pay for it."""
        if self.persistent:
            await self._ensure_connected()

    async def close(self) -> None:
        if self._loop is asyncio.get_running_loop():
            await self._discard(self._generation)

    async def _ensure_connected(self) -> tuple[Client, int]:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            # A session belongs to one event loop; start over if it changed (tests, scripts).
            self._loop, self._lock = loop, asyncio.Lock()
            self._client = self._owner = self._stop = None
        assert self._lock is not None
        async with self._lock:
            if self._client is None or self._owner is None or self._owner.done():
                await self._connect()
            assert self._client is not None
            return self._client, self._generation

    async def _connect(self) -> None:
        ready: asyncio.Future[Client] = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()
        self._stop = stop
        self._owner = asyncio.create_task(self._run(ready, stop), name="mcp-session-owner")
        try:
            self._client = await asyncio.wait_for(ready, self._connect_timeout)
        except BaseException:
            self._owner.cancel()
            self._client = self._owner = self._stop = None
            raise
        self._generation += 1

    async def _run(self, ready: asyncio.Future[Client], stop: asyncio.Event) -> None:
        try:
            async with Client(_server_parameters(), elicitation_callback=_on_elicitation) as client:
                if not ready.done():
                    ready.set_result(client)
                await stop.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            if not isinstance(error, asyncio.CancelledError):
                logger.warning("MCP session owner stopped: %s", type(error).__name__)
        finally:
            if self._stop is stop:
                self._client = None

    async def _discard(self, generation: int) -> None:
        """Stop the current connection, unless another caller already replaced it."""
        if self._lock is None:
            return
        async with self._lock:
            if generation != self._generation or self._owner is None:
                return
            owner, stop = self._owner, self._stop
            self._client = self._owner = self._stop = None
            if stop is not None:
                stop.set()
            try:
                await asyncio.wait_for(owner, 5)
            except (TimeoutError, asyncio.CancelledError, Exception):
                owner.cancel()


@lru_cache
def get_mcp_session() -> McpSession:
    return McpSession(persistent=get_settings().mcp_persistent_session)
