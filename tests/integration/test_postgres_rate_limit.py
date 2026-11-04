"""The Postgres rate limiter must count one user's requests across all workers."""

from __future__ import annotations

import asyncio
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from support_agents.db.connection import create_pool
from support_agents.limits import PostgresRateLimiter
from support_agents.settings import get_settings


def _database_reachable() -> bool:
    try:
        url = urlparse(get_settings().database_url)
        with socket.create_connection((url.hostname, url.port or 5432), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_reachable(), reason="PostgreSQL is not reachable")


def test_requests_are_counted_across_workers() -> None:
    async def scenario() -> None:
        user = f"test-{uuid4()}"
        pool_a, pool_b = await create_pool(), await create_pool()
        try:
            worker_a = PostgresRateLimiter(pool_a, limit=3, window_seconds=3600)
            worker_b = PostgresRateLimiter(pool_b, limit=3, window_seconds=3600)
            first = [
                await worker_a.check(user),
                await worker_b.check(user),
                await worker_a.check(user),
            ]
            assert all(decision.allowed for decision in first)
            blocked = await worker_b.check(user)
            assert blocked.allowed is False
            assert 1 <= blocked.retry_after_seconds <= 3600
            assert (await worker_b.check(f"other-{uuid4()}")).allowed is True
        finally:
            await pool_a.execute("DELETE FROM rate_limits WHERE user_id LIKE 'test-%'")
            await pool_a.close()
            await pool_b.close()

    asyncio.run(scenario())


def test_concurrent_requests_never_exceed_the_limit() -> None:
    async def scenario() -> None:
        user = f"test-{uuid4()}"
        pool = await create_pool()
        try:
            limiter = PostgresRateLimiter(pool, limit=5, window_seconds=3600)
            decisions = await asyncio.gather(*(limiter.check(user) for _ in range(12)))
            assert sum(d.allowed for d in decisions) == 5
        finally:
            await pool.execute("DELETE FROM rate_limits WHERE user_id LIKE 'test-%'")
            await pool.close()

    asyncio.run(scenario())
