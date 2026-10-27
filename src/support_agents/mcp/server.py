"""Read-only MCP server for retrieving fee schedules from PostgreSQL."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Annotated, Literal, cast

import asyncpg
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver import (
    AcceptedElicitation,
    Elicit,
    ElicitationResult,
    Resolve,
)
from pydantic import BaseModel, Field

from support_agents.db.connection import create_pool
from support_agents.embeddings import OllamaEmbedder, to_pgvector
from support_agents.schemas import (
    AcademicSearchMatch,
    AcademicSearchResult,
    FeeItem,
    FeeLookupResult,
)
from support_agents.settings import get_settings

_NUMERIC_STANDARD = re.compile(
    r"(?:(?:class|standard|grade)\s*)?(\d{1,2})", re.IGNORECASE
)
_BOOK_LANGUAGE = re.compile(r"\b(english|hindi)\b", re.IGNORECASE)
_REQUESTED_STANDARD = re.compile(r"\b(?:standard|class|grade)\s*(10|[1-9])\b", re.IGNORECASE)
_SUBJECT_ALIASES = (
    ("Integrated world learning", ("integrated world learning", "world around us")),
    ("Environmental studies", ("environmental studies", "environmental science", "evs")),
    ("Social science", ("social science", "social studies")),
    ("Arts and wellbeing", ("arts and wellbeing", "arts and well-being", "art")),
    ("Mathematics", ("mathematics", "maths", "math")),
    ("Science", ("science",)),
    ("English", ("english",)),
    ("Hindi", ("hindi",)),
)


def _requested_book_filters(query: str) -> tuple[str | None, str | None]:
    """Extract explicit grade/subject filters to keep book retrieval on-record."""
    grade_match = _REQUESTED_STANDARD.search(query)
    standard = f"Standard {int(grade_match.group(1))}" if grade_match else None

    # In phrases such as “Science book in English”, English is the requested
    # edition language, not the subject record to retrieve.
    subject_query = re.sub(
        r"\b(?:in|language|medium)\s+(english|hindi)\b", " ", query, flags=re.IGNORECASE
    )
    subject = None
    for canonical, aliases in _SUBJECT_ALIASES:
        if any(re.search(rf"\b{re.escape(alias)}\b", subject_query, re.IGNORECASE) for alias in aliases):
            subject = canonical
            break
    return standard, subject


def _is_exact_syllabus_scope_query(query: str) -> bool:
    """Recognize requests the indexed guide explicitly says it cannot define."""
    normalized = query.casefold()
    asks_for_syllabus = "syllabus" in normalized or "curriculum" in normalized
    asks_for_exact_scope = (
        ("school" in normalized and any(term in normalized for term in ("exact", "specific")))
        or "chapter-by-chapter" in normalized
        or ("official" in normalized and re.search(r"20\d{2}\s*[-–]\s*20\d{2}", normalized) is not None)
    )
    return asks_for_syllabus and asks_for_exact_scope


class BookLanguageChoice(BaseModel):
    language: Literal["English", "Hindi"] = Field(
        description="Language for the requested book recommendations."
    )


def resolve_book_language(
    query: str, document_type: str
) -> BookLanguageChoice | Elicit[BookLanguageChoice]:
    """Ask the user which supported language they want for book recommendations."""
    if document_type != "book_recommendations":
        return BookLanguageChoice(language="English")

    mentioned = {match.title() for match in _BOOK_LANGUAGE.findall(query)}
    if len(mentioned) == 1:
        return BookLanguageChoice(language=next(iter(mentioned)))

    return Elicit(
        "Which language should I use for the book recommendations? Choose English or Hindi.",
        BookLanguageChoice,
    )


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
    standard_match = _NUMERIC_STANDARD.fullmatch(normalized_standard)
    if standard_match:
        normalized_standard = f"Class {int(standard_match.group(1))}"
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
        candidate_rows = await pool.fetch(
            """
            SELECT DISTINCT board
            FROM fees
            WHERE lower(standard) = lower($2)
              AND academic_year = $3
              AND (
                    strpos(lower(board), lower($1)) > 0
                    OR strpos(lower($1), lower(board)) > 0
              )
            ORDER BY board
            LIMIT 5
            """,
            normalized_board,
            normalized_standard,
            academic_year,
            timeout=10,
        )
        board_candidates = [row["board"] for row in candidate_rows]
        message = "No fee record was found for the requested board, standard, and academic year."
        if board_candidates:
            message = (
                "No exact board match was found. Similar board records are available; "
                "ask the user to confirm a candidate before looking it up."
            )
        return FeeLookupResult(
            found=False,
            board=normalized_board,
            standard=normalized_standard,
            academic_year=academic_year,
            fee_items=[],
            total_amount=None,
            currency=None,
            billing_frequency=None,
            message=message,
            board_candidates=board_candidates,
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
    language: Annotated[
        ElicitationResult[BookLanguageChoice], Resolve(resolve_book_language)
    ],
    limit: Annotated[int, Field(ge=1, le=10, description="Maximum number of relevant passages")] = 4,
    document_type: Annotated[
        Literal["syllabus", "book_recommendations"],
        Field(description="Academic source type to search."),
    ] = "syllabus",
) -> AcademicSearchResult:
    """Find relevant passages from the selected ingested academic source."""
    if document_type == "book_recommendations":
        if not isinstance(language, AcceptedElicitation):
            return AcademicSearchResult(
                matches=[],
                message="The user cancelled the book-language question.",
                elicitation_cancelled=True,
            )
        selected_language = language.data.language
    else:
        selected_language = None

    normalized_query = query.strip()
    if not normalized_query:
        return AcademicSearchResult(matches=[], message="Search query must not be blank.")

    settings = get_settings()
    app_context = cast(AppContext, ctx.request_context.lifespan_context)
    search_query = normalized_query
    scope_query = document_type == "syllabus" and _is_exact_syllabus_scope_query(
        normalized_query
    )
    if scope_query:
        # Surface the guide's explicit scope caveat instead of ranking a grade's
        # subject list above the answer to an exact/school-specific syllabus ask.
        search_query += (
            " The source is a broad overview, not an official year-specific syllabus; "
            "it cannot establish the exact syllabus for a particular school."
        )
    if selected_language:
        search_query = f"{normalized_query}\nRequested book language: {selected_language}."
    vector = to_pgvector(await app_context.embedder.embed(search_query))
    requested_standard, requested_subject = (
        _requested_book_filters(normalized_query)
        if document_type == "book_recommendations"
        else (None, None)
    )
    rows = await app_context.pool.fetch(
        """
        SELECT content,
               metadata->>'source_title' AS source,
               NULLIF(metadata->>'page', '')::INTEGER AS page,
               NULLIF(metadata->>'page_end', '')::INTEGER AS page_end,
               metadata->>'standard' AS standard,
               metadata->>'subject' AS subject,
               1 - (embedding <=> $1::text::vector) AS similarity_score
        FROM academic_documents
        WHERE embedding_model = $2
          AND COALESCE(metadata->>'document_type', 'syllabus') = $3
          AND ($5::text IS NULL OR metadata->>'standard' = $5)
          AND ($6::text IS NULL OR lower(metadata->>'subject') = lower($6))
          AND (
                NOT $7::boolean
                OR content ILIKE '%not an official year-specific syllabus%'
                OR content ILIKE '%cannot establish the exact subjects or syllabus for a particular school%'
              )
        ORDER BY embedding <=> $1::text::vector
        LIMIT $4
        """,
        vector,
        app_context.embedder.model,
        document_type,
        limit,
        requested_standard,
        requested_subject,
        scope_query,
        timeout=10,
    )
    matches = [
        AcademicSearchMatch(
            content=row["content"],
            source=row["source"] or "Academic subject guide",
            page=row["page"],
            page_end=row["page_end"],
            standard=row["standard"],
            subject=row["subject"],
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
    return AcademicSearchResult(
        matches=matches,
        message=message,
        language=selected_language,
    )


if __name__ == "__main__":
    settings = get_settings()
    if settings.mcp_transport == "stdio":
        mcp.run()
    elif settings.mcp_transport == "streamable-http":
        mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)
    else:
        raise ValueError("MCP_TRANSPORT must be 'stdio' or 'streamable-http'.")
