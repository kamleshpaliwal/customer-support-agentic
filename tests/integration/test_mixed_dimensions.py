"""One table holds vectors of different sizes; searches only ever touch their own model."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from support_agents.db.connection import create_pool
from support_agents.embeddings import to_pgvector
from support_agents.retrieval import SearchFilters, vector_candidates
from tests.integration.test_incremental_ingest import pytestmark  # noqa: F401


async def _insert(pool, source_id, model, content, vector):
    await pool.execute(
        "INSERT INTO academic_documents (source_id, chunk_index, content, metadata, "
        "embedding_model, embedding) VALUES ($1, 0, $2, '{}'::jsonb, $3, $4::text::vector)",
        source_id, content, model, to_pgvector(vector),
    )


def test_each_model_only_sees_its_own_vectors():
    async def scenario():
        pool = await create_pool()
        small, large = f"test-768-{uuid4()}", f"test-1024-{uuid4()}"
        ids = [f"{uuid4().hex:0<64}"[:64], f"{uuid4().hex:0<64}"[:64]]
        try:
            await _insert(pool, ids[0], small, "from the small model", [0.1] * 768)
            await _insert(pool, ids[1], large, "from the large model", [0.1] * 1024)

            for model, size, expected in ((small, 768, "small"), (large, 1024, "large")):
                found = await vector_candidates(
                    pool,
                    vector=to_pgvector([0.1] * size),
                    filters=SearchFilters(model, "syllabus", None, None, False),
                    limit=5,
                )
                assert [m.content for m in found] == [f"from the {expected} model"]
        finally:
            await pool.execute(
                "DELETE FROM academic_documents WHERE source_id = ANY($1::bpchar[])", ids
            )
            await pool.close()

    asyncio.run(scenario())
