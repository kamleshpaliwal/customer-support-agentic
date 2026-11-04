"""Concurrency limiter and in-memory rate limiter behavior."""

from __future__ import annotations

import asyncio

import pytest

from support_agents.limits import ConcurrencyLimiter, InMemoryRateLimiter, ServiceBusyError


def test_limits_simultaneous_calls_and_rejects_after_the_wait() -> None:
    async def scenario() -> tuple[int, int]:
        limiter = ConcurrencyLimiter(max_concurrent=2, queue_timeout_seconds=0.1)
        running = peak = 0
        release = asyncio.Event()

        async def call() -> None:
            nonlocal running, peak
            async with limiter.slot():
                running += 1
                peak = max(peak, running)
                await release.wait()
                running -= 1

        holders = [asyncio.create_task(call()) for _ in range(2)]
        await asyncio.sleep(0.02)
        with pytest.raises(ServiceBusyError):
            async with limiter.slot():
                pytest.fail("third caller must not get a slot")
        release.set()
        await asyncio.gather(*holders)
        return peak, running

    assert asyncio.run(scenario()) == (2, 0)


def test_waiting_caller_gets_the_slot_when_one_frees_up() -> None:
    async def scenario() -> list[str]:
        limiter = ConcurrencyLimiter(max_concurrent=1, queue_timeout_seconds=2)
        order: list[str] = []

        async def first() -> None:
            async with limiter.slot():
                order.append("first")
                await asyncio.sleep(0.05)

        async def second() -> None:
            await asyncio.sleep(0.01)
            async with limiter.slot():
                order.append("second")

        await asyncio.gather(first(), second())
        return order

    assert asyncio.run(scenario()) == ["first", "second"]


def test_slot_is_released_when_the_call_fails() -> None:
    async def scenario() -> None:
        limiter = ConcurrencyLimiter(max_concurrent=1, queue_timeout_seconds=0.05)
        with pytest.raises(RuntimeError):
            async with limiter.slot():
                raise RuntimeError("model failed")
        async with limiter.slot():  # would raise ServiceBusyError if the slot leaked
            pass

    asyncio.run(scenario())


def test_zero_means_unlimited_and_limiter_survives_new_event_loops() -> None:
    async def many() -> None:
        unlimited = ConcurrencyLimiter(max_concurrent=0, queue_timeout_seconds=0.01)
        async with unlimited.slot(), unlimited.slot(), unlimited.slot():
            pass

    asyncio.run(many())

    limiter = ConcurrencyLimiter(max_concurrent=1, queue_timeout_seconds=0.05)

    async def use() -> None:
        async with limiter.slot():
            pass

    asyncio.run(use())
    asyncio.run(use())  # a different loop must not trip over the old semaphore


def test_rate_limiter_allows_the_limit_then_blocks_with_retry_after() -> None:
    now = [100.0]
    limiter = InMemoryRateLimiter(limit=3, window_seconds=60, clock=lambda: now[0])

    async def scenario():
        allowed = [(await limiter.check("alice")).allowed for _ in range(3)]
        blocked = await limiter.check("alice")
        other_user = await limiter.check("bob")
        now[0] += 60  # next window
        reset = await limiter.check("alice")
        return allowed, blocked, other_user, reset

    allowed, blocked, other_user, reset = asyncio.run(scenario())
    assert allowed == [True, True, True]
    assert blocked.allowed is False and 1 <= blocked.retry_after_seconds <= 61
    assert other_user.allowed is True  # limits are per user
    assert reset.allowed is True
