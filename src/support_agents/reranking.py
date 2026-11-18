"""Second-stage reranking with a cross-encoder (local) or the Amazon Bedrock Rerank API.

A bi-encoder (the embedding search) scores a passage without seeing the query. A cross-encoder reads
the query and the passage together, which is more accurate but cannot be precomputed, so it is only
run on the few candidates the first stage returns.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Protocol

from support_agents.aws import create_aws_client
from support_agents.settings import Settings, get_settings


class Reranker(Protocol):
    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage; higher is more relevant."""
        ...

    async def warm_up(self) -> None:
        """Prepare the model so the first query does not pay for it."""
        ...


class CrossEncoderReranker:
    """Local cross-encoder (``sentence-transformers``). The model loads on first use."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._load_lock = asyncio.Lock()

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:  # pragma: no cover - depends on the optional extra
                raise RuntimeError(
                    "Reranking needs the optional dependency: pip install -e '.[rerank]'"
                ) from error
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def warm_up(self) -> None:
        async with self._load_lock:
            await asyncio.to_thread(self._load)

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        await self.warm_up()
        return await asyncio.to_thread(self._predict, query, passages)

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        scores = self._model.predict(
            [(query, passage) for passage in passages], show_progress_bar=False
        )
        return [float(value) for value in scores]


class RerankError(RuntimeError):
    """Raised when the Bedrock Rerank API cannot score the passages."""


class BedrockReranker:
    """Amazon Bedrock Rerank API (``bedrock-agent-runtime``); no model runs in this process."""

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        active = settings or get_settings()
        self.model_id = active.rerank_bedrock_model_id
        self.region = active.rerank_bedrock_region
        self.model_arn = f"arn:aws:bedrock:{self.region}::foundation-model/{self.model_id}"
        self._client = client or create_aws_client(
            "bedrock-agent-runtime", active, region_name=self.region, config=_client_config()
        )

    async def warm_up(self) -> None:
        return None

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return await asyncio.to_thread(self._rerank, query, passages)

    def _rerank(self, query: str, passages: list[str]) -> list[float]:
        try:
            response = self._client.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": query}}],
                sources=[
                    {
                        "type": "INLINE",
                        "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": text}},
                    }
                    for text in passages
                ],
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {"modelArn": self.model_arn},
                        "numberOfResults": len(passages),
                    },
                },
            )
        except Exception as error:
            raise RerankError(f"Bedrock rerank failed for model {self.model_id!r}.") from error

        # Results come back sorted by score, each pointing at its input position.
        scores: list[float | None] = [None] * len(passages)
        for result in response.get("results", []):
            index = result.get("index")
            if isinstance(index, int) and 0 <= index < len(passages):
                scores[index] = float(result["relevanceScore"])
        if any(score is None for score in scores):
            raise RerankError("Bedrock rerank did not score every passage.")
        return [float(score) for score in scores if score is not None]


def _client_config() -> Any:
    from botocore.config import Config

    # Reranking is optional: fail fast under throttling and let the caller keep the first-stage
    # order, rather than making the user wait through long retries.
    return Config(
        connect_timeout=5,
        read_timeout=30,
        retries={"max_attempts": 3, "mode": "standard"},
    )


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    if settings.rerank_provider == "bedrock":
        return BedrockReranker(settings)
    return CrossEncoderReranker(settings.academic_reranker_model)
