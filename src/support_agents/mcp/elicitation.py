"""Bridge MCP elicitation prompts to the browser-based support UI.

A tool call holds one request open while the user answers through two other requests (poll the
question, submit the answer). With several API workers those requests can land on different
processes, so the PostgreSQL broker keeps the pending question in a shared table. The in-memory
broker is for tests and single-process development.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import asyncpg
from mcp.types import ElicitResult

DEFAULT_TTL_SECONDS = 300.0


@dataclass
class PendingElicitation:
    request_id: str
    message: str
    field_name: str
    options: list[str]


class ElicitationBroker(Protocol):
    async def ask(
        self, request_id: str, *, message: str, field_name: str, options: list[str]
    ) -> ElicitResult: ...

    async def get_pending(self, request_id: str) -> PendingElicitation | None: ...

    async def submit(self, request_id: str, answer: str) -> str: ...


def _resolve_answer(options: list[str], answer: str) -> tuple[str, str | None]:
    """Return ("cancel", None) or ("accept", option); raise ValueError for other answers."""
    normalized = answer.strip()
    if normalized.casefold() in {"cancel", "skip"}:
        return "cancel", None
    selected = next(
        (option for option in options if option.casefold() == normalized.casefold()), None
    )
    if selected is None:
        raise ValueError("Choose one of the listed options or cancel the question.")
    return "accept", selected


def _result(action: str, field_name: str, answer: str | None) -> ElicitResult:
    if action != "accept" or answer is None:
        return ElicitResult(action="cancel")
    return ElicitResult(action="accept", content={field_name: answer})


class InMemoryElicitationBroker:
    """Single-process broker. State is lost on restart and not shared between workers."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingElicitation] = {}
        self._futures: dict[str, asyncio.Future[tuple[str, str | None]]] = {}

    async def ask(
        self, request_id: str, *, message: str, field_name: str, options: list[str]
    ) -> ElicitResult:
        if request_id in self._pending:
            raise RuntimeError("An elicitation is already pending for this request.")
        future: asyncio.Future[tuple[str, str | None]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = PendingElicitation(request_id, message, field_name, options)
        self._futures[request_id] = future
        try:
            action, answer = await future
            return _result(action, field_name, answer)
        finally:
            self._pending.pop(request_id, None)
            self._futures.pop(request_id, None)

    async def get_pending(self, request_id: str) -> PendingElicitation | None:
        return self._pending.get(request_id)

    async def submit(self, request_id: str, answer: str) -> str:
        pending = self._pending.get(request_id)
        future = self._futures.get(request_id)
        if pending is None or future is None or future.done():
            raise LookupError("No MCP elicitation is pending for this request.")
        action, selected = _resolve_answer(pending.options, answer)
        future.set_result((action, selected))
        return "cancel" if action == "cancel" else "accepted"


class PostgresElicitationBroker:
    """Shared broker: the pending question lives in ``pending_elicitations``.

    ``ask`` polls its row until another request (on any worker) records an answer or the
    question expires.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        poll_interval_seconds: float = 0.3,
    ) -> None:
        self._pool = pool
        self._ttl_seconds = ttl_seconds
        self._poll_interval_seconds = poll_interval_seconds

    async def ask(
        self, request_id: str, *, message: str, field_name: str, options: list[str]
    ) -> ElicitResult:
        key = UUID(request_id)
        async with self._pool.acquire() as connection:
            # Drop a stale row from an earlier ask with the same id before inserting.
            await connection.execute(
                "DELETE FROM pending_elicitations "
                "WHERE request_id = $1 AND (status <> 'pending' OR expires_at <= now())",
                key,
            )
            try:
                await connection.execute(
                    "INSERT INTO pending_elicitations "
                    "(request_id, message, field_name, options, expires_at) "
                    "VALUES ($1, $2, $3, $4::jsonb, now() + make_interval(secs => $5))",
                    key,
                    message,
                    field_name,
                    json.dumps(options),
                    self._ttl_seconds,
                )
            except asyncpg.UniqueViolationError as error:
                raise RuntimeError("An elicitation is already pending for this request.") from error

        try:
            while True:
                row = await self._pool.fetchrow(
                    "SELECT status, answer, expires_at <= now() AS expired "
                    "FROM pending_elicitations WHERE request_id = $1",
                    key,
                )
                if row is None or row["status"] == "cancelled" or row["expired"]:
                    return ElicitResult(action="cancel")
                if row["status"] == "accepted":
                    return _result("accept", field_name, row["answer"])
                await asyncio.sleep(self._poll_interval_seconds)
        finally:
            # Shielded so the row is removed even if this request is cancelled.
            await asyncio.shield(
                self._pool.execute("DELETE FROM pending_elicitations WHERE request_id = $1", key)
            )

    async def get_pending(self, request_id: str) -> PendingElicitation | None:
        row = await self._pool.fetchrow(
            "SELECT message, field_name, options FROM pending_elicitations "
            "WHERE request_id = $1 AND status = 'pending' AND expires_at > now()",
            UUID(request_id),
        )
        if row is None:
            return None
        return PendingElicitation(
            request_id, row["message"], row["field_name"], json.loads(row["options"])
        )

    async def submit(self, request_id: str, answer: str) -> str:
        async with self._pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT options FROM pending_elicitations "
                "WHERE request_id = $1 AND status = 'pending' AND expires_at > now() "
                "FOR UPDATE",
                UUID(request_id),
            )
            if row is None:
                raise LookupError("No MCP elicitation is pending for this request.")
            action, selected = _resolve_answer(json.loads(row["options"]), answer)
            await connection.execute(
                "UPDATE pending_elicitations SET status = $2, answer = $3 WHERE request_id = $1",
                UUID(request_id),
                "cancelled" if action == "cancel" else "accepted",
                selected,
            )
            return "cancel" if action == "cancel" else "accepted"


_broker: ElicitationBroker = InMemoryElicitationBroker()


def get_elicitation_broker() -> ElicitationBroker:
    return _broker


def set_elicitation_broker(broker: ElicitationBroker) -> None:
    global _broker
    _broker = broker
