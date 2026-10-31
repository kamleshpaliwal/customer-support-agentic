"""Which user started which request, so one user cannot resume or answer another's session."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

import asyncpg


class RequestOwnership(Protocol):
    async def claim(self, request_id: str, user_id: str) -> bool:
        """Record the owner; True if newly claimed or already owned by this user."""
        ...

    async def is_owner(self, request_id: str, user_id: str) -> bool: ...


class InMemoryRequestOwnership:
    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    async def claim(self, request_id: str, user_id: str) -> bool:
        return self._owners.setdefault(request_id, user_id) == user_id

    async def is_owner(self, request_id: str, user_id: str) -> bool:
        return self._owners.get(request_id) == user_id


class PostgresRequestOwnership:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def claim(self, request_id: str, user_id: str) -> bool:
        owner = await self._pool.fetchval(
            "INSERT INTO request_owners (request_id, user_id) VALUES ($1, $2) "
            "ON CONFLICT (request_id) DO UPDATE SET request_id = EXCLUDED.request_id "
            "RETURNING user_id",
            UUID(request_id),
            user_id,
        )
        return owner == user_id

    async def is_owner(self, request_id: str, user_id: str) -> bool:
        owner = await self._pool.fetchval(
            "SELECT user_id FROM request_owners WHERE request_id = $1", UUID(request_id)
        )
        return owner == user_id
