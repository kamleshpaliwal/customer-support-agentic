"""The academic agent's chat model is chosen by ACADEMIC_AGENT_PROVIDER."""

from __future__ import annotations

import asyncio

import pytest
from langchain_aws import ChatBedrockConverse
from langchain_ollama import ChatOllama

from support_agents.agents import academic_specialist as academic_module
from support_agents.agents.academic_specialist import create_academic_chat_model
from support_agents.limits import ConcurrencyLimiter, ServiceBusyError
from support_agents.settings import Settings


def _settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        aws_access_key_id="AKIATESTKEY",
        aws_secret_access_key="test-secret",
        **overrides,
    )


def test_ollama_is_the_default_provider():
    model = create_academic_chat_model(
        _settings(academic_agent_ollama_model="local-model", ollama_base_url="http://ollama:11434")
    )
    assert isinstance(model, ChatOllama)
    assert model.model == "local-model"
    assert model.base_url == "http://ollama:11434"


def test_bedrock_provider_builds_a_converse_model_from_settings():
    model = create_academic_chat_model(
        _settings(
            academic_agent_provider="bedrock",
            academic_agent_bedrock_model_id="us.amazon.nova-pro-v1:0",
            aws_region="eu-west-1",
        )
    )
    assert isinstance(model, ChatBedrockConverse)
    assert model.model_id == "us.amazon.nova-pro-v1:0"
    assert model.region_name == "eu-west-1"


def test_bedrock_model_supports_the_agents_tools():
    from langchain_core.tools import tool

    @tool
    def search_syllabus(query: str) -> str:
        """Search the syllabus."""
        return query

    model = create_academic_chat_model(_settings(academic_agent_provider="bedrock"))
    assert model.bind_tools([search_syllabus]) is not None


def test_bedrock_model_waits_for_a_slot_around_each_call(monkeypatch: pytest.MonkeyPatch):
    limiter = ConcurrencyLimiter(max_concurrent=1, queue_timeout_seconds=0.05)
    monkeypatch.setattr(academic_module, "get_model_limiter", lambda: limiter)
    calls: list[str] = []

    async def fake_agenerate(self, *args, **kwargs):
        calls.append("called")
        return "result"

    monkeypatch.setattr(ChatBedrockConverse, "_agenerate", fake_agenerate)
    model = create_academic_chat_model(_settings(academic_agent_provider="bedrock"))

    async def scenario() -> None:
        assert await model._agenerate([]) == "result"
        async with limiter.slot():
            with pytest.raises(ServiceBusyError):
                await model._agenerate([])
        assert calls == ["called"]

    asyncio.run(scenario())
