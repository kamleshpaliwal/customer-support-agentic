"""Local Ollama embedding client shared by document ingestion and MCP retrieval."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any
from urllib.request import Request, urlopen

from support_agents.settings import Settings, get_settings


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot return a valid embedding vector."""


EMBEDDING_DIMENSIONS = 768


class OllamaEmbedder:
    """Create 768-dimensional vectors using the configured local Ollama model."""

    def __init__(self, settings: Settings | None = None) -> None:
        active = settings or get_settings()
        self.model = active.ollama_embedding_model
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


def to_pgvector(vector: list[float]) -> str:
    """Serialize a validated vector for PostgreSQL's vector text input format."""
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise EmbeddingError(f"Expected a {EMBEDDING_DIMENSIONS}-dimensional vector.")
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"
