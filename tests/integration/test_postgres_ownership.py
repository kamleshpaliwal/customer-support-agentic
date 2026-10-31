"""Request ownership persisted in PostgreSQL is shared by every worker."""

from __future__ import annotations

import asyncio
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from support_agents.db.connection import create_pool
from support_agents.ownership import PostgresRequestOwnership
from support_agents.settings import get_settings


def _database_reachable() -> bool:
    try:
        url = urlparse(get_settings().database_url)
        with socket.create_connection((url.hostname, url.port or 5432), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_reachable(), reason="PostgreSQL is not reachable")


def test_first_claim_wins_across_workers() -> None:
    async def scenario() -> None:
        request_id = str(uuid4())
        pool_a, pool_b = await create_pool(), await create_pool()
        try:
            worker_a, worker_b = PostgresRequestOwnership(pool_a), PostgresRequestOwnership(pool_b)
            assert await worker_a.claim(request_id, "alice") is True
            assert await worker_a.claim(request_id, "alice") is True  # idempotent for the owner
            assert await worker_b.claim(request_id, "bob") is False  # cannot hijack
            assert await worker_b.is_owner(request_id, "alice") is True
            assert await worker_b.is_owner(request_id, "bob") is False
            assert await worker_b.is_owner(str(uuid4()), "alice") is False
        finally:
            await pool_a.close()
            await pool_b.close()

    asyncio.run(scenario())
