"""The Postgres broker must work when ask and answer happen on different 'workers'."""

from __future__ import annotations

import asyncio
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from support_agents.db.connection import create_pool
from support_agents.mcp.elicitation import PostgresElicitationBroker
from support_agents.settings import get_settings


def _database_reachable() -> bool:
    try:
        url = urlparse(get_settings().database_url)
        with socket.create_connection((url.hostname, url.port or 5432), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_reachable(), reason="PostgreSQL is not reachable")

OPTIONS = ["English", "Hindi"]


async def _wait_pending(broker, request_id: str):
    for _ in range(100):
        pending = await broker.get_pending(request_id)
        if pending is not None:
            return pending
        await asyncio.sleep(0.02)
    raise AssertionError("elicitation never became pending")


async def _scenario(answer: str):
    request_id = str(uuid4())
    # Separate pools stand in for separate worker processes.
    pool_a, pool_b = await create_pool(), await create_pool()
    try:
        worker_a = PostgresElicitationBroker(pool_a, poll_interval_seconds=0.02)
        worker_b = PostgresElicitationBroker(pool_b, poll_interval_seconds=0.02)
        asking = asyncio.create_task(
            worker_a.ask(request_id, message="Language?", field_name="language", options=OPTIONS)
        )
        pending = await _wait_pending(worker_b, request_id)
        assert (pending.message, pending.options) == ("Language?", OPTIONS)
        with pytest.raises(ValueError):
            await worker_b.submit(request_id, "French")
        await worker_b.submit(request_id, answer)
        result = await asking
        assert await worker_b.get_pending(request_id) is None
        with pytest.raises(LookupError):
            await worker_b.submit(request_id, "English")
        return result
    finally:
        await pool_a.close()
        await pool_b.close()


def test_answer_from_another_worker_reaches_the_waiting_request() -> None:
    result = asyncio.run(_scenario("hindi"))
    assert result.action == "accept"
    assert result.content == {"language": "Hindi"}


def test_cancel_from_another_worker() -> None:
    assert asyncio.run(_scenario("cancel")).action == "cancel"


def test_unanswered_question_expires() -> None:
    async def scenario():
        pool = await create_pool()
        try:
            broker = PostgresElicitationBroker(pool, ttl_seconds=0.3, poll_interval_seconds=0.05)
            return await broker.ask(
                str(uuid4()), message="Language?", field_name="language", options=OPTIONS
            )
        finally:
            await pool.close()

    assert asyncio.run(scenario()).action == "cancel"
