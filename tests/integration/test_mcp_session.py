"""The persistent MCP session against the real MCP server process (needs PostgreSQL)."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import time
from urllib.parse import urlparse

import pytest
from mcp.types import ElicitResult

from support_agents.mcp.elicitation import get_elicitation_broker, set_elicitation_broker
from support_agents.mcp.session import McpSession
from support_agents.settings import get_settings


def _database_reachable() -> bool:
    try:
        url = urlparse(get_settings().database_url)
        with socket.create_connection((url.hostname, url.port or 5432), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_reachable(), reason="PostgreSQL is not reachable")

FEE_ARGS = {"board": "DEMO-CBSE", "standard": "Class 5", "academic_year": "2026-2027"}


def server_pids() -> list[int]:
    """MCP server processes started by this test process."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, check=True
    ).stdout
    return [
        int(line.split()[0])
        for line in out.splitlines()
        if "support_agents.mcp.server" in line and int(line.split()[1]) == os.getpid()
    ]


async def wait_for_exit(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while server_pids() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


def test_many_calls_share_one_server_process() -> None:
    async def scenario() -> None:
        session = McpSession()
        try:
            for _ in range(3):
                assert not (await session.call_tool("get_fees", FEE_ARGS)).is_error
            results = await asyncio.gather(
                *(session.call_tool("get_fees", FEE_ARGS) for _ in range(10))
            )
            assert all(not r.is_error and r.structured_content for r in results)
            assert len(server_pids()) == 1
        finally:
            await session.close()
        await wait_for_exit()
        assert server_pids() == []

    asyncio.run(scenario())


def test_calls_after_the_first_are_much_faster_than_starting_a_process() -> None:
    async def scenario() -> tuple[float, float]:
        session = McpSession()
        try:
            await session.call_tool("get_fees", FEE_ARGS)  # pays the one-time connect
            started = time.perf_counter()
            for _ in range(5):
                await session.call_tool("get_fees", FEE_ARGS)
            warm = (time.perf_counter() - started) / 5
        finally:
            await session.close()
        spawning = McpSession(persistent=False)
        started = time.perf_counter()
        await spawning.call_tool("get_fees", FEE_ARGS)
        return warm, time.perf_counter() - started

    warm, spawn = asyncio.run(scenario())
    assert warm < spawn / 3, f"warm call {warm:.3f}s vs spawn {spawn:.3f}s"


def test_recovers_when_the_server_process_dies() -> None:
    async def scenario() -> None:
        session = McpSession()
        try:
            await session.call_tool("get_fees", FEE_ARGS)
            (old_pid,) = server_pids()
            os.kill(old_pid, signal.SIGKILL)
            await asyncio.sleep(0.3)

            result = await session.call_tool("get_fees", FEE_ARGS)  # reconnects and retries
            assert not result.is_error
            (new_pid,) = server_pids()
            assert new_pid != old_pid
        finally:
            await session.close()

    asyncio.run(scenario())


def test_non_persistent_mode_starts_a_process_per_call_and_leaves_none() -> None:
    async def scenario() -> None:
        session = McpSession(persistent=False)
        for _ in range(2):
            assert not (await session.call_tool("get_fees", FEE_ARGS)).is_error
            await wait_for_exit()
            assert server_pids() == []

    asyncio.run(scenario())


class RecordingBroker:
    """Stands in for the browser: records which request each prompt belongs to, then cancels."""

    def __init__(self) -> None:
        self.asked_for: list[str] = []

    async def ask(self, request_id, *, message, field_name, options):
        self.asked_for.append(request_id)
        await asyncio.sleep(0.2)  # long enough for both prompts to be pending together
        return ElicitResult(action="cancel")

    async def get_pending(self, request_id):
        return None

    async def submit(self, request_id, answer):
        raise LookupError


def test_elicitation_prompts_are_routed_to_the_right_request() -> None:
    previous = get_elicitation_broker()
    broker = RecordingBroker()
    set_elicitation_broker(broker)

    async def scenario() -> None:
        session = McpSession()
        args = {"limit": 2, "document_type": "book_recommendations"}
        try:
            results = await asyncio.gather(
                session.call_tool(
                    "search_academic_documents",
                    {"query": "Books for Standard 5 Mathematics", **args},
                    request_id="request-A",
                ),
                session.call_tool(
                    "search_academic_documents",
                    {"query": "Books for Standard 3 Science", **args},
                    request_id="request-B",
                ),
            )
            assert all(not r.is_error for r in results)
            assert sorted(broker.asked_for) == ["request-A", "request-B"]
        finally:
            await session.close()

    try:
        asyncio.run(scenario())
    finally:
        set_elicitation_broker(previous)


def test_prompt_without_a_request_id_is_declined_not_misrouted() -> None:
    previous = get_elicitation_broker()
    broker = RecordingBroker()
    set_elicitation_broker(broker)

    async def scenario() -> None:
        session = McpSession()
        try:
            result = await session.call_tool(
                "search_academic_documents",
                {
                    "query": "Books for Standard 5 Mathematics",
                    "document_type": "book_recommendations",
                },
            )
            assert not result.is_error
            assert broker.asked_for == []  # nobody to ask, so nobody was asked
        finally:
            await session.close()

    try:
        asyncio.run(scenario())
    finally:
        set_elicitation_broker(previous)
