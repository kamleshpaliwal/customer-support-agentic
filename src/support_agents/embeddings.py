"""Embedding clients (local Ollama or Amazon Bedrock) shared by ingestion and MCP retrieval."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Protocol
from urllib.request import Request, urlopen

from support_agents.aws import create_bedrock_runtime_client
from support_agents.settings import Settings, get_settings


class EmbeddingError(RuntimeError):
    """Raised when the embedding provider cannot return a valid embedding vector."""


EMBEDDING_DIMENSIONS = 768  # the local Ollama model (nomic-embed-text)


class Embedder(Protocol):
    """``model`` identifies the vector space: vectors are only comparable within one model."""

    model: str
    dimensions: int

    async def embed(self, text: str) -> list[float]: ...

    async def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Create 768-dimensional vectors using the configured local Ollama model."""

    def __init__(self, settings: Settings | None = None) -> None:
        active = settings or get_settings()
        self.model = active.ollama_embedding_model
        self.dimensions = EMBEDDING_DIMENSIONS
        self.base_url = active.ollama_base_url.rstrip("/")
        self.timeout_seconds = active.ollama_request_timeout_seconds

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed, text)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_many, texts)

    def _embed(self, text: str) -> list[float]:
        return self._embed_many([text])[0]

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise EmbeddingError("Embedding input must contain non-empty text.")
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body: dict[str, Any] = json.loads(response.read())
        except Exception as error:
            raise EmbeddingError(
                f"Could not generate embeddings using local Ollama model {self.model!r}."
            ) from error

        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected number of embeddings.")
        result: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
                raise EmbeddingError(
                    f"Ollama model {self.model!r} must return {EMBEDDING_DIMENSIONS} dimensions."
                )
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError) as error:
                raise EmbeddingError("Ollama returned a non-numeric embedding value.") from error
            if not all(math.isfinite(value) for value in values):
                raise EmbeddingError("Ollama returned a non-finite embedding value.")
            result.append(values)
        return result


class BedrockEmbedder:
    """Amazon Titan Text Embeddings V2 through the Bedrock runtime (one text per request)."""

    MAX_CONCURRENT_REQUESTS = 4

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        active = settings or get_settings()
        self.model_id = active.embedding_bedrock_model_id
        self.dimensions = active.embedding_bedrock_dimensions
        # The stored model name includes the size, so two sizes of one model never mix.
        self.model = f"{self.model_id}@{self.dimensions}"
        self._client = client or create_bedrock_runtime_client(
            active,
            config=_bedrock_config(),
        )

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise EmbeddingError("Embedding input must contain non-empty text.")
        limit = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        async def one(text: str) -> list[float]:
            async with limit:
                return await asyncio.to_thread(self._embed_one, text)

        return list(await asyncio.gather(*(one(text) for text in texts)))

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps(
            {"inputText": text, "dimensions": self.dimensions, "normalize": True}
        )
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
        except Exception as error:
            raise EmbeddingError(
                f"Could not generate embeddings using Bedrock model {self.model_id!r}."
            ) from error
        return _validated_vector(payload.get("embedding"), self.dimensions, self.model_id)


def _bedrock_config() -> Any:
    from botocore.config import Config

    return Config(
        connect_timeout=5,
        read_timeout=30,
        retries={"max_attempts": 3, "mode": "standard"},
    )


def _validated_vector(vector: Any, dimensions: int, model: str) -> list[float]:
    if not isinstance(vector, list) or len(vector) != dimensions:
        raise EmbeddingError(f"Model {model!r} must return {dimensions} dimensions.")
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as error:
        raise EmbeddingError("The embedding contains a non-numeric value.") from error
    if not all(math.isfinite(value) for value in values):
        raise EmbeddingError("The embedding contains a non-finite value.")
    return values


def create_embedder(settings: Settings | None = None) -> Embedder:
    """Create the configured embedder; local Ollama is the development default."""
    active = settings or get_settings()
    if active.embedding_provider == "ollama":
        return OllamaEmbedder(active)
    return BedrockEmbedder(active)


def to_pgvector(vector: list[float]) -> str:
    """Serialize a validated vector for PostgreSQL's vector text input format."""
    if not vector:
        raise EmbeddingError("Cannot serialize an empty vector.")
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"
