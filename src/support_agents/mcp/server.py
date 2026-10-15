"""Read-only MCP server for retrieving fee schedules from PostgreSQL."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, cast

import asyncpg
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from support_agents.db.connection import create_pool
from support_agents.embeddings import OllamaEmbedder, to_pgvector
from support_agents.schemas import (
    AcademicSearchMatch,
    AcademicSearchResult,
    FeeItem,
    FeeLookupResult,
)
from support_agents.settings import get_settings

@dataclass
class AppContext:
    pool: asyncpg.Pool
    embedder: OllamaEmbedder


@asynccontextmanager
async def server_lifespan(_: MCPServer) -> AsyncIterator[AppContext]:
    pool = await create_pool()
    try:
        yield AppContext(pool=pool, embedder=OllamaEmbedder())
    finally:
        await pool.close()


mcp = MCPServer(
    "support-fees",
    instructions="Read-only fee lookup by education board, standard, and academic year.",
    lifespan=server_lifespan,
)


@mcp.tool()
async def get_fees(
    board: Annotated[
        str, Field(min_length=1, max_length=50, description="Education board, e.g. CBSE")
    ],
    standard: Annotated[
        str, Field(min_length=1, max_length=30, description="Standard, e.g. Class 10")
    ],
    academic_year: Annotated[
        str,
        Field(
            pattern=r"^[0-9]{4}-[0-9]{4}$",
            description="Academic year in YYYY-YYYY format, e.g. 2026-2027",
        ),
    ],
    ctx: Context,
) -> FeeLookupResult:
    """Look up recorded fee components for a board, standard, and academic year."""

    normalized_board = board.strip()
    normalized_standard = standard.strip()
    if not normalized_board or not normalized_standard:
        return FeeLookupResult(
            found=False,
            board=board,
            standard=standard,
            academic_year=academic_year,
            fee_items=[],
            total_amount=None,
            currency=None,
            billing_frequency=None,
            message="Board and standard must contain non-whitespace characters.",
        )
    start_year, end_year = (int(part) for part in academic_year.split("-"))
    if end_year != start_year + 1:
        return FeeLookupResult(
            found=False,
            board=normalized_board,
            standard=normalized_standard,
            academic_year=academic_year,
            fee_items=[],
            total_amount=None,
            currency=None,
            billing_frequency=None,
            message="Academic year must contain consecutive years, such as 2026-2027.",
        )

    app_context = cast(AppContext, ctx.request_context.lifespan_context)
    pool = app_context.pool
    rows = await pool.fetch(
        """
        SELECT fee_type, amount, currency, billing_frequency
        FROM fees
        WHERE lower(board) = lower($1)
          AND lower(standard) = lower($2)
          AND academic_year = $3
        ORDER BY fee_type
        """,
        normalized_board,
        normalized_standard,
        academic_year,
        timeout=10,
    )

    if not rows:
        return FeeLookupResult(
            found=False,
            board=normalized_board,
            standard=normalized_standard,
            academic_year=academic_year,
            fee_items=[],
            total_amount=None,
            currency=None,
            billing_frequency=None,
            message="No fee record was found for the requested board, standard, and academic year.",
        )

    currencies = {row["currency"].strip() for row in rows}
    items = [
        FeeItem(
            fee_type=row["fee_type"],
            amount=format(Decimal(row["amount"]), ".2f"),
            currency=row["currency"].strip(),
            billing_frequency=row["billing_frequency"],
        )
        for row in rows
    ]
    frequencies = {row["billing_frequency"] for row in rows}
    if len(currencies) != 1 or len(frequencies) != 1:
        return FeeLookupResult(
            found=True,
            board=normalized_board,
            standard=normalized_standard,
            academic_year=academic_year,
            fee_items=items,
            total_amount=None,
            currency=None,
            billing_frequency=None,
            message=(
                "Fee components use different currencies or billing frequencies; "
                "a combined total cannot be calculated."
            ),
        )

    currency = currencies.pop()
    total = sum((Decimal(row["amount"]) for row in rows), start=Decimal("0.00"))
    return FeeLookupResult(
        found=True,
        board=normalized_board,
        standard=normalized_standard,
        academic_year=academic_year,
        fee_items=items,
        total_amount=format(total, ".2f"),
        currency=currency,
        billing_frequency=frequencies.pop(),
        message="Fee records found.",
    )


@mcp.tool()
async def search_academic_documents(
    query: Annotated[
        str,
        Field(min_length=1, max_length=2000, description="Academic question to search for"),
    ],
    ctx: Context,
    limit: Annotated[int, Field(ge=1, le=10, description="Maximum number of relevant passages")] = 4,
) -> AcademicSearchResult:
    """Find relevant passages from the ingested academic subject guide."""
    normalized_query = query.strip()
    if not normalized_query:
        return AcademicSearchResult(matches=[], message="Search query must not be blank.")

    settings = get_settings()
    app_context = cast(AppContext, ctx.request_context.lifespan_context)
    vector = to_pgvector(await app_context.embedder.embed(normalized_query))
    rows = await app_context.pool.fetch(
        """
        SELECT content,
               metadata->>'source_title' AS source,
               NULLIF(metadata->>'page', '')::INTEGER AS page,
               metadata->>'standard' AS standard,
               1 - (embedding <=> $1::text::vector) AS similarity_score
        FROM academic_documents
        WHERE embedding_model = $2
        ORDER BY embedding <=> $1::text::vector
        LIMIT $3
        """,
        vector,
        app_context.embedder.model,
        limit,
        timeout=10,
    )
    matches = [
        AcademicSearchMatch(
            content=row["content"],
            source=row["source"] or "Academic subject guide",
            page=row["page"],
            standard=row["standard"],
            similarity_score=min(1.0, max(-1.0, float(row["similarity_score"]))),
        )
        for row in rows
        if float(row["similarity_score"]) >= settings.academic_search_min_similarity
    ]
    message = (
        "Relevant passages found."
        if matches
        else "No sufficiently relevant academic passage was found in the indexed guide."
    )
    return AcademicSearchResult(matches=matches, message=message)


if __name__ == "__main__":
    settings = get_settings()
    if settings.mcp_transport == "stdio":
        mcp.run()
    elif settings.mcp_transport == "streamable-http":
        mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)
    else:
        raise ValueError("MCP_TRANSPORT must be 'stdio' or 'streamable-http'.")
