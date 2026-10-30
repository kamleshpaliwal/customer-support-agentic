"""In-memory elicitation broker behavior (the Postgres broker is covered in integration)."""

from __future__ import annotations

import asyncio

import pytest

from support_agents.mcp.elicitation import InMemoryElicitationBroker

OPTIONS = ["English", "Hindi"]


async def _ask(broker: InMemoryElicitationBroker, request_id: str = "r1"):
    return await broker.ask(request_id, message="Language?", field_name="language", options=OPTIONS)


async def _wait_pending(broker: InMemoryElicitationBroker, request_id: str = "r1") -> None:
    for _ in range(100):
        if await broker.get_pending(request_id) is not None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("elicitation never became pending")


def test_accepts_a_listed_option_case_insensitively() -> None:
    async def scenario():
        broker = InMemoryElicitationBroker()
        task = asyncio.create_task(_ask(broker))
        await _wait_pending(broker)
        pending = await broker.get_pending("r1")
        assert pending is not None and pending.options == OPTIONS
        assert await broker.submit("r1", "hindi") == "accepted"
        return await task

    result = asyncio.run(scenario())
    assert result.action == "accept"
    assert result.content == {"language": "Hindi"}


def test_cancel_and_invalid_answers() -> None:
    async def scenario():
        broker = InMemoryElicitationBroker()
        task = asyncio.create_task(_ask(broker))
        await _wait_pending(broker)
        with pytest.raises(ValueError):
            await broker.submit("r1", "French")
        assert await broker.submit("r1", "skip") == "cancel"
        result = await task
        assert await broker.get_pending("r1") is None
        with pytest.raises(LookupError):
            await broker.submit("r1", "English")
        return result

    assert asyncio.run(scenario()).action == "cancel"
