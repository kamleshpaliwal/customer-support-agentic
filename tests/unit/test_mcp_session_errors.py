"""Which errors mean the MCP server connection is gone (reconnect) vs a normal tool error."""

from __future__ import annotations

import anyio
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, INTERNAL_ERROR, INVALID_PARAMS

from support_agents.mcp.session import is_connection_lost


def test_connection_failures_trigger_reconnect() -> None:
    assert is_connection_lost(MCPError(CONNECTION_CLOSED, "Connection closed"))
    assert is_connection_lost(anyio.ClosedResourceError())
    assert is_connection_lost(anyio.BrokenResourceError())
    assert is_connection_lost(BrokenPipeError())


def test_ordinary_errors_do_not_trigger_reconnect() -> None:
    assert not is_connection_lost(MCPError(INVALID_PARAMS, "bad arguments"))
    assert not is_connection_lost(MCPError(INTERNAL_ERROR, "tool crashed"))
    assert not is_connection_lost(ValueError("anything else"))
    assert not is_connection_lost(TimeoutError())
