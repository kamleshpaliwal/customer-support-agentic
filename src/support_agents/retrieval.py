"""Passage retrieval over the indexed academic documents.

Retrieval is split into stages so each can be measured on its own (see ``evals/retrieval_eval.py``):

1. candidates: ``vector_candidates`` (embedding similarity) and, in hybrid mode,
   ``keyword_candidates`` (PostgreSQL full-text search), merged with reciprocal rank fusion;
2. optional rerank: a cross-encoder reorders the candidates (see ``reranking.py``);
3. the final cut to ``limit`` passages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

import asyncpg

from support_agents.embeddings import OllamaEmbedder, to_pgvector
from support_agents.reranking import Reranker, get_reranker
from support_agents.schemas import AcademicSearchMatch
from support_agents.settings import get_settings

logger = logging.getLogger(__name__)

DocumentType = Literal["syllabus", "book_recommendations"]
RetrievalMode = Literal["vector", "hybrid"]
RRF_K = 60  # damps the weight of top ranks; 60 is the value from the original RRF paper

_REQUESTED_STANDARD = re.compile(r"\b(?:standard|class|grade)\s*(10|[1-9])\b", re.IGNORECASE)
_SCOPE_CAVEAT_PATTERNS = (
    "%not an official year-specific syllabus%",
    "%cannot establish the exact subjects or syllabus for a particular school%",
)
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


def requested_book_filters(query: str) -> tuple[str | None, str | None]:
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
        if any(
            re.search(rf"\b{re.escape(alias)}\b", subject_query, re.IGNORECASE)
            for alias in aliases
        ):
            subject = canonical
            break
    return standard, subject


def is_exact_syllabus_scope_query(query: str) -> bool:
    """Recognize requests the indexed guide explicitly says it cannot define."""
    normalized = query.casefold()
    asks_for_syllabus = "syllabus" in normalized or "curriculum" in normalized
    asks_for_exact_scope = (
        ("school" in normalized and any(term in normalized for term in ("exact", "specific")))
        or "chapter-by-chapter" in normalized
        or (
            "official" in normalized
            and re.search(r"20\d{2}\s*[-–]\s*20\d{2}", normalized) is not None
        )
    )
    return asks_for_syllabus and asks_for_exact_scope


def build_search_query(query: str, document_type: DocumentType, language: str | None) -> str:
    """Add the context the embedding needs: scope caveat for syllabus asks, book language."""
    search_query = query
    if document_type == "syllabus" and is_exact_syllabus_scope_query(query):
        # Surface the guide's explicit scope caveat instead of ranking a grade's
        # subject list above the answer to an exact/school-specific syllabus ask.
        search_query += (
            " The source is a broad overview, not an official year-specific syllabus; "
            "it cannot establish the exact syllabus for a particular school."
        )
    if language:
        search_query = f"{query}\nRequested book language: {language}."
    return search_query


@dataclass(frozen=True)
class SearchFilters:
    embedding_model: str
    document_type: DocumentType
    standard: str | None
    subject: str | None
    scope_query: bool


# Shared by both candidate queries, so the same chunks are eligible whichever way they are found.
_SELECT = """
    SELECT content,
           metadata->>'source_title' AS source,
           NULLIF(metadata->>'page', '')::INTEGER AS page,
           NULLIF(metadata->>'page_end', '')::INTEGER AS page_end,
           metadata->>'standard' AS standard,
           metadata->>'subject' AS subject,
           1 - (embedding <=> $1::text::vector) AS similarity_score
"""
_WHERE = """
    WHERE embedding_model = $2
      AND COALESCE(metadata->>'document_type', 'syllabus') = $3
      AND ($5::text IS NULL OR metadata->>'standard' = $5)
      AND ($6::text IS NULL OR lower(metadata->>'subject') = lower($6))
      AND (NOT $7::boolean OR content ILIKE ANY($8::text[]))
"""
_VECTOR_SQL = (
    _SELECT
    + "FROM academic_documents"
    + _WHERE
    + "ORDER BY embedding <=> $1::text::vector LIMIT $4"
)
_KEYWORD_SQL = (
    _SELECT
    + "FROM academic_documents, to_tsquery('english', $9) AS q"
    + _WHERE
    + "AND content_tsv @@ q ORDER BY ts_rank_cd(content_tsv, q) DESC LIMIT $4"
)


def _to_match(row: asyncpg.Record) -> AcademicSearchMatch:
    return AcademicSearchMatch(
        content=row["content"],
        source=row["source"] or "Academic subject guide",
        page=row["page"],
        page_end=row["page_end"],
        standard=row["standard"],
        subject=row["subject"],
        similarity_score=min(1.0, max(-1.0, float(row["similarity_score"]))),
    )


def _filter_args(filters: SearchFilters, vector: str, limit: int) -> tuple[object, ...]:
    return (
        vector,
        filters.embedding_model,
        filters.document_type,
        limit,
        filters.standard,
        filters.subject,
        filters.scope_query,
        list(_SCOPE_CAVEAT_PATTERNS),
    )


async def vector_candidates(
    pool: asyncpg.Pool, *, vector: str, filters: SearchFilters, limit: int
) -> list[AcademicSearchMatch]:
    """Nearest chunks by cosine similarity (bi-encoder), after metadata filters."""
    rows = await pool.fetch(_VECTOR_SQL, *_filter_args(filters, vector, limit), timeout=10)
    return [_to_match(row) for row in rows]


def keyword_tsquery(query: str) -> str | None:
    """OR together the query's words for ``to_tsquery`` (stop words and stemming are applied there).

    OR rather than AND because user questions are sentences: requiring every word to appear would
    match nothing. ``ts_rank_cd`` then ranks chunks that contain more of the words higher.
    """
    # Keep digits even when single ("Standard 6"): they are often the most selective term.
    words = dict.fromkeys(
        word
        for word in re.findall(r"\w+", query.casefold())
        if len(word) > 1 or word.isdigit()
    )
    return " | ".join(words) or None


async def keyword_candidates(
    pool: asyncpg.Pool, *, query: str, vector: str, filters: SearchFilters, limit: int
) -> list[AcademicSearchMatch]:
    """Chunks sharing words with the query, ranked by full-text relevance.

    ``similarity_score`` is still the cosine similarity so one threshold applies to every candidate.
    """
    tsquery = keyword_tsquery(query)
    if tsquery is None:
        return []
    rows = await pool.fetch(
        _KEYWORD_SQL, *_filter_args(filters, vector, limit), tsquery, timeout=10
    )
    return [_to_match(row) for row in rows]


def _identity(match: AcademicSearchMatch) -> tuple[object, ...]:
    return (match.content, match.page, match.page_end, match.standard, match.subject)


def reciprocal_rank_fusion(
    rankings: list[list[AcademicSearchMatch]], *, k: int = RRF_K
) -> list[AcademicSearchMatch]:
    """Merge ranked lists: each appearance adds 1 / (k + rank), so agreement between lists wins.

    Only ranks are used, so the lists' scores (cosine vs. ts_rank) never need to be comparable.
    """
    scores: dict[tuple[object, ...], float] = {}
    matches: dict[tuple[object, ...], AcademicSearchMatch] = {}
    for ranking in rankings:
        for rank, match in enumerate(ranking, start=1):
            key = _identity(match)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            matches.setdefault(key, match)
    return [matches[key] for key in sorted(scores, key=lambda key: scores[key], reverse=True)]


def order_by_scores(
    candidates: list[AcademicSearchMatch], scores: list[float]
) -> list[AcademicSearchMatch]:
    """Reorder candidates by reranker score, highest first (ties keep their earlier order)."""
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [match for match, _ in ranked]


async def search_passages(
    pool: asyncpg.Pool,
    embedder: OllamaEmbedder,
    *,
    query: str,
    document_type: DocumentType,
    language: str | None = None,
    limit: int = 4,
    mode: RetrievalMode | None = None,
    rerank: bool | None = None,
    reranker: Reranker | None = None,
) -> list[AcademicSearchMatch]:
    """Return the final passages for a (non-blank) query.

    ``mode``, ``rerank`` and ``reranker`` default to the settings; the eval passes them explicitly
    to compare strategies on the same data.
    """
    settings = get_settings()
    mode = mode or settings.academic_retrieval_mode
    rerank = settings.academic_rerank_enabled if rerank is None else rerank
    normalized = query.strip()
    vector = to_pgvector(
        await embedder.embed(build_search_query(normalized, document_type, language))
    )
    standard, subject = (
        requested_book_filters(normalized)
        if document_type == "book_recommendations"
        else (None, None)
    )
    filters = SearchFilters(
        embedding_model=embedder.model,
        document_type=document_type,
        standard=standard,
        subject=subject,
        scope_query=document_type == "syllabus" and is_exact_syllabus_scope_query(normalized),
    )

    # Fetch a wider pool only when a later stage will narrow it back down.
    wide = mode == "hybrid" or rerank
    pool_size = max(limit, settings.academic_candidate_pool) if wide else limit
    candidates = await vector_candidates(pool, vector=vector, filters=filters, limit=pool_size)
    if mode == "hybrid":
        keyword = await keyword_candidates(
            pool, query=normalized, vector=vector, filters=filters, limit=pool_size
        )
        candidates = reciprocal_rank_fusion([candidates, keyword])[:pool_size]

    min_similarity = settings.academic_search_min_similarity
    candidates = [match for match in candidates if match.similarity_score >= min_similarity]
    if rerank and len(candidates) > 1:
        try:
            scores = await (reranker or get_reranker()).score(
                normalized, [match.content for match in candidates]
            )
            candidates = order_by_scores(candidates, scores)
        except Exception:
            logger.exception("Reranking failed; keeping first-stage order.")
    return candidates[:limit]
