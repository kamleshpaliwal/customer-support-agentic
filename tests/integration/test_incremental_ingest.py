"""Incremental ingestion against PostgreSQL with a counting fake embedder."""

from __future__ import annotations

import asyncio
import hashlib
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from support_agents.db.connection import create_pool
from support_agents.db.ingest_academic_pdf import DocumentChunk, chunk_hash, ingest_chunks
from support_agents.embeddings import EMBEDDING_DIMENSIONS
from support_agents.settings import get_settings


def _database_reachable() -> bool:
    try:
        url = urlparse(get_settings().database_url)
        with socket.create_connection((url.hostname, url.port or 5432), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_reachable(), reason="PostgreSQL is not reachable")


class FakeEmbedder:
    def __init__(self, model: str = "fake-model-a") -> None:
        self.model = model
        self.embedded_texts: list[str] = []

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = hashlib.sha256(text.encode()).digest()
        return [seed[i % len(seed)] / 255 + 0.001 for i in range(EMBEDDING_DIMENSIONS)]


def _chunks(*texts: str, page: int = 1) -> list[DocumentChunk]:
    return [DocumentChunk(text, page=page, standard="Standard 1") for text in texts]


async def _run(scenario) -> None:
    pool = await create_pool()
    source_id = f"test-{uuid4()}"
    try:
        await scenario(pool, source_id)
    finally:
        await pool.execute("DELETE FROM academic_documents WHERE source_id = $1", source_id)
        await pool.close()


async def _ingest(pool, source_id, chunks, embedder, **kwargs):
    return await ingest_chunks(
        chunks,
        source_id=source_id,
        document_type="syllabus",
        source_title="Test source",
        source_file="test.pdf",
        embedder=embedder,
        pool=pool,
        **kwargs,
    )


async def _rows(pool, source_id):
    return await pool.fetch(
        "SELECT id, chunk_index, content, content_hash, embedding_model, "
        "metadata->>'page' AS page, embedding::text AS embedding "
        "FROM academic_documents WHERE source_id = $1 ORDER BY chunk_index",
        source_id,
    )


def test_unchanged_reingest_embeds_nothing_and_keeps_rows() -> None:
    async def scenario(pool, source_id):
        embedder = FakeEmbedder()
        first = await _ingest(pool, source_id, _chunks("a", "b", "c"), embedder)
        assert (first.total, first.embedded, first.reused, first.removed) == (3, 3, 0, 0)
        before = await _rows(pool, source_id)

        embedder.embedded_texts.clear()
        second = await _ingest(pool, source_id, _chunks("a", "b", "c"), embedder)
        assert (second.embedded, second.reused, second.removed) == (0, 3, 0)
        assert embedder.embedded_texts == []
        assert [r["id"] for r in await _rows(pool, source_id)] == [r["id"] for r in before]

    asyncio.run(_run(scenario))


def test_edit_embeds_only_the_changed_chunk() -> None:
    async def scenario(pool, source_id):
        embedder = FakeEmbedder()
        await _ingest(pool, source_id, _chunks("a", "b", "c"), embedder)
        before = {r["content"]: r for r in await _rows(pool, source_id)}

        embedder.embedded_texts.clear()
        result = await _ingest(pool, source_id, _chunks("a", "B edited", "c"), embedder)
        assert embedder.embedded_texts == ["B edited"]
        assert (result.embedded, result.reused, result.removed) == (1, 2, 1)

        after = {r["content"]: r for r in await _rows(pool, source_id)}
        assert set(after) == {"a", "B edited", "c"}
        for text in ("a", "c"):  # untouched chunks keep their row and vector
            assert after[text]["id"] == before[text]["id"]
            assert after[text]["embedding"] == before[text]["embedding"]

    asyncio.run(_run(scenario))


def test_insert_in_middle_renumbers_without_unique_violation() -> None:
    async def scenario(pool, source_id):
        embedder = FakeEmbedder()
        await _ingest(pool, source_id, _chunks("a", "b", "c"), embedder)
        ids = {r["content"]: r["id"] for r in await _rows(pool, source_id)}

        embedder.embedded_texts.clear()
        await _ingest(pool, source_id, _chunks("a", "new", "b", "c"), embedder)
        assert embedder.embedded_texts == ["new"]

        rows = await _rows(pool, source_id)
        assert [(r["chunk_index"], r["content"]) for r in rows] == [
            (0, "a"), (1, "new"), (2, "b"), (3, "c")
        ]
        assert {r["content"]: r["id"] for r in rows if r["content"] != "new"} == ids

    asyncio.run(_run(scenario))


def test_swapped_chunks_are_reindexed_and_metadata_is_refreshed() -> None:
    async def scenario(pool, source_id):
        embedder = FakeEmbedder()
        await _ingest(pool, source_id, _chunks("a", "b", page=1), embedder)
        embedder.embedded_texts.clear()
        await _ingest(pool, source_id, _chunks("b", "a", page=7), embedder)
        assert embedder.embedded_texts == []
        rows = await _rows(pool, source_id)
        assert [r["content"] for r in rows] == ["b", "a"]
        assert {r["page"] for r in rows} == {"7"}  # page moved, text did not

    asyncio.run(_run(scenario))


def test_removed_chunk_is_deleted() -> None:
    async def scenario(pool, source_id):
        embedder = FakeEmbedder()
        await _ingest(pool, source_id, _chunks("a", "b", "c"), embedder)
        embedder.embedded_texts.clear()
        result = await _ingest(pool, source_id, _chunks("a", "c"), embedder)
        assert (result.embedded, result.reused, result.removed) == (0, 2, 1)
        assert [r["content"] for r in await _rows(pool, source_id)] == ["a", "c"]

    asyncio.run(_run(scenario))


def test_model_change_and_force_reembed_everything() -> None:
    async def scenario(pool, source_id):
        await _ingest(pool, source_id, _chunks("a", "b"), FakeEmbedder("model-a"))

        other = FakeEmbedder("model-b")
        result = await _ingest(pool, source_id, _chunks("a", "b"), other)
        assert result.embedded == 2 and other.embedded_texts == ["a", "b"]
        assert {r["embedding_model"] for r in await _rows(pool, source_id)} == {"model-b"}

        forced = await _ingest(pool, source_id, _chunks("a", "b"), other, force=True)
        assert forced.embedded == 2 and forced.reused == 0

    asyncio.run(_run(scenario))


def test_rows_store_the_content_hash() -> None:
    async def scenario(pool, source_id):
        await _ingest(pool, source_id, _chunks("hello"), FakeEmbedder())
        (row,) = await _rows(pool, source_id)
        assert row["content_hash"] == chunk_hash("hello")

    asyncio.run(_run(scenario))


def test_concurrent_ingests_of_one_source_do_not_duplicate() -> None:
    async def scenario(pool, source_id):
        embedder = FakeEmbedder()
        await asyncio.gather(
            *(_ingest(pool, source_id, _chunks("a", "b", "c"), embedder) for _ in range(3))
        )
        rows = await _rows(pool, source_id)
        assert [r["chunk_index"] for r in rows] == [0, 1, 2]
        assert [r["content"] for r in rows] == ["a", "b", "c"]

    asyncio.run(_run(scenario))
