"""Keyword (full-text) candidates against PostgreSQL, using synthetic chunks."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from support_agents.db.connection import create_pool
from support_agents.db.ingest_academic_pdf import DocumentChunk, ingest_chunks
from support_agents.embeddings import to_pgvector
from support_agents.retrieval import SearchFilters, keyword_candidates, vector_candidates
from tests.integration.test_incremental_ingest import FakeEmbedder, pytestmark  # noqa: F401

CHUNKS = [
    DocumentChunk("Photosynthesis happens in chloroplasts.", page=1, standard="Standard 7"),
    DocumentChunk("Fractions and decimals are taught early.", page=2, standard="Standard 7"),
    DocumentChunk("The monsoon shapes the climate of India.", page=3, standard="Standard 7"),
]


def test_keyword_search_finds_exact_words_and_respects_filters():
    async def scenario():
        pool = await create_pool()
        source_id = f"test-{uuid4()}"
        embedder = FakeEmbedder(model=f"fake-{uuid4()}")
        try:
            await ingest_chunks(
                CHUNKS,
                source_id=source_id,
                document_type="syllabus",
                source_title="Test source",
                source_file="test.pdf",
                embedder=embedder,
                pool=pool,
            )
            vector = to_pgvector((await embedder.embed_many(["anything"]))[0])
            filters = SearchFilters(embedder.model, "syllabus", None, None, False)

            found = await keyword_candidates(
                pool, query="Where does photosynthesis happen?", vector=vector,
                filters=filters, limit=5,
            )
            assert [m.page for m in found] == [1]  # only the chunk sharing a word

            assert await keyword_candidates(
                pool, query="?!", vector=vector, filters=filters, limit=5
            ) == []

            other_standard = SearchFilters(embedder.model, "syllabus", "Standard 9", None, False)
            assert await keyword_candidates(
                pool, query="photosynthesis", vector=vector, filters=other_standard, limit=5
            ) == []

            everything = await vector_candidates(pool, vector=vector, filters=filters, limit=5)
            assert len(everything) == len(CHUNKS)
        finally:
            await pool.execute("DELETE FROM academic_documents WHERE source_id = $1", source_id)
            await pool.close()

    asyncio.run(scenario())
