"""Bedrock (Titan V2) embedder behaviour, using a fake runtime client."""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from support_agents.embeddings import (
    BedrockEmbedder,
    EmbeddingError,
    OllamaEmbedder,
    create_embedder,
    to_pgvector,
)
from support_agents.mcp.session import _server_parameters
from support_agents.settings import Settings


def settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        aws_access_key_id="AKIATESTKEY",
        aws_secret_access_key="test-secret",
        **overrides,
    )


class FakeRuntime:
    def __init__(self, dimensions: int = 1024, fail: bool = False) -> None:
        self.dimensions, self.fail, self.calls = dimensions, fail, []

    def invoke_model(self, *, modelId, body, contentType, accept):
        self.calls.append({"modelId": modelId, "body": json.loads(body)})
        if self.fail:
            raise RuntimeError("AccessDenied")
        seed = len(self.calls[-1]["body"]["inputText"]) / 100
        payload = {"embedding": [seed] * self.dimensions, "inputTextTokenCount": 3}
        return {"body": io.BytesIO(json.dumps(payload).encode())}


def test_titan_request_and_model_identity():
    fake = FakeRuntime()
    embedder = BedrockEmbedder(settings(), client=fake)
    vector = asyncio.run(embedder.embed("hello"))
    assert len(vector) == 1024
    assert fake.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    assert fake.calls[0]["body"] == {"inputText": "hello", "dimensions": 1024, "normalize": True}
    assert embedder.model == "amazon.titan-embed-text-v2:0@1024"
    assert embedder.dimensions == 1024


def test_the_stored_model_name_includes_the_dimension():
    smaller = BedrockEmbedder(settings(embedding_bedrock_dimensions=512), client=FakeRuntime(512))
    assert smaller.model == "amazon.titan-embed-text-v2:0@512"


def test_embed_many_keeps_the_input_order():
    fake = FakeRuntime()
    vectors = asyncio.run(
        BedrockEmbedder(settings(), client=fake).embed_many(["a", "bb", "cccc"])
    )
    assert [v[0] for v in vectors] == [0.01, 0.02, 0.04]


def test_wrong_dimension_is_rejected():
    embedder = BedrockEmbedder(settings(), client=FakeRuntime(dimensions=768))
    with pytest.raises(EmbeddingError, match="1024 dimensions"):
        asyncio.run(embedder.embed("hello"))


def test_blank_input_is_rejected():
    embedder = BedrockEmbedder(settings(), client=FakeRuntime())
    with pytest.raises(EmbeddingError):
        asyncio.run(embedder.embed_many(["ok", "  "]))


def test_provider_errors_become_embedding_errors():
    embedder = BedrockEmbedder(settings(), client=FakeRuntime(fail=True))
    with pytest.raises(EmbeddingError, match="Bedrock model"):
        asyncio.run(embedder.embed("hello"))


def test_factory_follows_the_provider_setting():
    assert isinstance(create_embedder(settings()), OllamaEmbedder)
    assert isinstance(create_embedder(settings(embedding_provider="bedrock")), BedrockEmbedder)


def test_pgvector_accepts_any_dimension():
    assert to_pgvector([0.5, 1.0]) == "[0.5,1]"
    with pytest.raises(EmbeddingError):
        to_pgvector([])


def test_the_mcp_server_receives_aws_and_embedding_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/abc")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setenv("UNRELATED_SECRET", "nope")
    env = _server_parameters().env
    assert env["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"] == "/v2/credentials/abc"
    assert env["EMBEDDING_PROVIDER"] == "bedrock"
    assert "UNRELATED_SECRET" not in env


def test_dimensions_load_from_an_environment_string(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_BEDROCK_DIMENSIONS", "512")
    loaded = Settings(_env_file=None, database_url="postgresql://unused/unused")
    assert loaded.embedding_bedrock_dimensions == 512


def test_unsupported_dimensions_are_rejected():
    with pytest.raises(ValueError, match="256, 512 or 1024"):
        settings(embedding_bedrock_dimensions=768)
