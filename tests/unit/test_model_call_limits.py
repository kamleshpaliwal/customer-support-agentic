"""Busy errors must reach the API, not be swallowed by the model adapters."""

from __future__ import annotations

import asyncio

import pytest

from support_agents.agents import supervisor as supervisor_module
from support_agents.agents.supervisor import OllamaIntentDetector
from support_agents.limits import ConcurrencyLimiter, ServiceBusyError


def test_supervisor_reports_busy_instead_of_a_generic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = ConcurrencyLimiter(max_concurrent=1, queue_timeout_seconds=0.05)
    monkeypatch.setattr(supervisor_module, "get_model_limiter", lambda: limiter)

    async def scenario() -> None:
        detector = OllamaIntentDetector()
        async with limiter.slot():  # the only slot is taken by another request
            with pytest.raises(ServiceBusyError):
                await detector.detect("What is the tuition fee?")

    asyncio.run(scenario())


def test_academic_agent_model_waits_for_a_slot_around_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_ollama import ChatOllama

    from support_agents.agents import academic_specialist as academic_module

    limiter = ConcurrencyLimiter(max_concurrent=1, queue_timeout_seconds=0.05)
    monkeypatch.setattr(academic_module, "get_model_limiter", lambda: limiter)
    calls: list[str] = []

    async def fake_agenerate(self, *args, **kwargs):
        calls.append("called")
        return "result"

    monkeypatch.setattr(ChatOllama, "_agenerate", fake_agenerate)
    model = academic_module._LimitedChatOllama(model="test-model")

    async def scenario() -> None:
        assert await model._agenerate([]) == "result"  # free slot: the call goes through
        async with limiter.slot():
            with pytest.raises(ServiceBusyError):
                await model._agenerate([])  # slot taken: busy, and the model is not called
        assert calls == ["called"]

    asyncio.run(scenario())
