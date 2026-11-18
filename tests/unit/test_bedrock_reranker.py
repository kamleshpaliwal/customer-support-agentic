"""Bedrock Rerank API adapter, using a fake bedrock-agent-runtime client."""

from __future__ import annotations

import asyncio

import pytest

from support_agents import reranking
from support_agents.reranking import BedrockReranker, CrossEncoderReranker, RerankError
from support_agents.settings import Settings


def settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        aws_access_key_id="AKIATESTKEY",
        aws_secret_access_key="test-secret",
        **overrides,
    )


class FakeAgentRuntime:
    def __init__(self, results=None, fail=False):
        self.results, self.fail, self.requests = results, fail, []

    def rerank(self, **request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("AccessDenied")
        if self.results is not None:
            return {"results": self.results}
        # Highest score to the last passage, returned sorted by score like the real API.
        count = len(request["sources"])
        ranked = sorted(range(count), key=lambda i: -i)
        return {"results": [{"index": i, "relevanceScore": (i + 1) / count} for i in ranked]}


def test_request_shape_and_model_arn():
    fake = FakeAgentRuntime()
    reranker = BedrockReranker(settings(rerank_bedrock_region="us-west-2"), client=fake)
    asyncio.run(reranker.score("what subjects", ["a", "b"]))
    request = fake.requests[0]
    assert request["queries"][0]["textQuery"]["text"] == "what subjects"
    texts = [s["inlineDocumentSource"]["textDocument"]["text"] for s in request["sources"]]
    assert texts == ["a", "b"]
    config = request["rerankingConfiguration"]["bedrockRerankingConfiguration"]
    assert config["modelConfiguration"]["modelArn"] == (
        "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
    )
    assert config["numberOfResults"] == 2


def test_scores_are_returned_in_input_order_not_result_order():
    fake = FakeAgentRuntime()
    scores = asyncio.run(BedrockReranker(settings(), client=fake).score("q", ["a", "b", "c"]))
    assert scores == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_a_missing_score_is_an_error():
    fake = FakeAgentRuntime(results=[{"index": 0, "relevanceScore": 0.9}])
    with pytest.raises(RerankError, match="every passage"):
        asyncio.run(BedrockReranker(settings(), client=fake).score("q", ["a", "b"]))


def test_provider_errors_become_rerank_errors():
    with pytest.raises(RerankError, match="amazon.rerank-v1:0"):
        failing = BedrockReranker(settings(), client=FakeAgentRuntime(fail=True))
        asyncio.run(failing.score("q", ["a"]))


def test_no_passages_makes_no_request():
    fake = FakeAgentRuntime()
    assert asyncio.run(BedrockReranker(settings(), client=fake).score("q", [])) == []
    assert fake.requests == []


def test_factory_follows_the_provider_setting(monkeypatch: pytest.MonkeyPatch):
    reranking.get_reranker.cache_clear()
    monkeypatch.setattr(reranking, "get_settings", lambda: settings(rerank_provider="bedrock"))
    assert isinstance(reranking.get_reranker(), BedrockReranker)
    reranking.get_reranker.cache_clear()
    monkeypatch.setattr(reranking, "get_settings", lambda: settings())
    assert isinstance(reranking.get_reranker(), CrossEncoderReranker)
    reranking.get_reranker.cache_clear()
