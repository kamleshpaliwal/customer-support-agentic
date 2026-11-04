"""Overload protection: a cap on simultaneous model calls and a per-user request rate limit.

* ``ConcurrencyLimiter`` keeps at most N model calls running at once. Extra callers wait up
  to a timeout, then get ``ServiceBusyError`` instead of making everyone slower.
* ``RateLimiter`` allows each user a number of requests per time window. The PostgreSQL
  implementation is shared by all API workers.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import asyncpg

from support_agents.settings import get_settings


class ServiceBusyError(RuntimeError):
    """The model backend is at capacity and the caller waited too long for a free slot."""


class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int, queue_timeout_seconds: float) -> None:
        self.max_concurrent = max_concurrent
        self.queue_timeout_seconds = queue_timeout_seconds
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        # A semaphore belongs to one event loop; rebuild if the loop changed (tests, scripts).
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._loop, self._semaphore = loop, asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        if self.max_concurrent <= 0:
            yield
            return
        semaphore = self._get_semaphore()
        try:
            async with asyncio.timeout(self.queue_timeout_seconds):
                await semaphore.acquire()
        except TimeoutError as error:
            raise ServiceBusyError("All model slots are busy.") from error
        try:
            yield
        finally:
            semaphore.release()


@lru_cache
def get_model_limiter() -> ConcurrencyLimiter:
    settings = get_settings()
    return ConcurrencyLimiter(settings.model_max_concurrency, settings.model_queue_timeout_seconds)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class RateLimiter(Protocol):
    async def check(self, user_id: str) -> RateLimitDecision:
        """Count one request for the user and say whether it is within the limit."""
        ...


class InMemoryRateLimiter:
    """Fixed-window limiter for one process (tests and local development)."""

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._windows: dict[str, tuple[int, int]] = {}  # user -> (window number, count)

    async def check(self, user_id: str) -> RateLimitDecision:
        now = self._clock()
        number = int(now // self._window)
        stored_number, count = self._windows.get(user_id, (number, 0))
        count = count + 1 if stored_number == number else 1
        self._windows[user_id] = (number, count)
        if count <= self._limit:
            return RateLimitDecision(True)
        return RateLimitDecision(False, max(1, int((number + 1) * self._window - now) + 1))


class PostgresRateLimiter:
    """Fixed-window limiter backed by ``rate_limits`` so every worker shares the counts.

    A fixed window can allow up to twice the limit across a window boundary; that is an
    accepted trade-off for one cheap upsert per request.
    """

    def __init__(self, pool: asyncpg.Pool, limit: int, window_seconds: int) -> None:
        self._pool = pool
        self._limit = limit
        self._window = window_seconds

    async def check(self, user_id: str) -> RateLimitDecision:
        row = await self._pool.fetchrow(
            """
            INSERT INTO rate_limits (user_id, window_start, request_count)
            VALUES ($1, date_bin(make_interval(secs => $2), now(), TIMESTAMPTZ 'epoch'), 1)
            ON CONFLICT (user_id, window_start)
                DO UPDATE SET request_count = rate_limits.request_count + 1
            RETURNING request_count,
                      CEIL(EXTRACT(EPOCH FROM (
                          window_start + make_interval(secs => $2) - now()
                      )))::int AS retry_after
            """,
            user_id,
            float(self._window),
        )
        if random.random() < 0.01:  # occasional cleanup of old windows
            await self._pool.execute(
                "DELETE FROM rate_limits WHERE window_start < now() - interval '1 day'"
            )
        if row["request_count"] <= self._limit:
            return RateLimitDecision(True)
        return RateLimitDecision(False, max(1, row["retry_after"]))
