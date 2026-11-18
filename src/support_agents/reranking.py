"""Second-stage reranking with a cross-encoder.

A bi-encoder (the embedding search) scores a passage without seeing the query. A cross-encoder reads
the query and the passage together, which is more accurate but cannot be precomputed, so it is only
run on the few candidates the first stage returns.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Protocol

from support_agents.settings import get_settings


class Reranker(Protocol):
    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage; higher is more relevant."""
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


@lru_cache
def get_reranker() -> Reranker:
    return CrossEncoderReranker(get_settings().academic_reranker_model)
