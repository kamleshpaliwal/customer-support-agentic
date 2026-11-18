"""Ingest configured academic PDFs into pgvector.

Run with: python -m support_agents.db.ingest_academic_pdf [path/to/document.pdf]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from pypdf import PdfReader

from support_agents.db.connection import create_pool
from support_agents.embeddings import create_embedder, to_pgvector
from support_agents.settings import PROJECT_ROOT, Settings, get_settings


GRADE_HEADING = re.compile(r"^Standard\s+(\d{1,2})$", re.IGNORECASE)
BOOK_HEADING = re.compile(
    r"^Standard\s+(?P<standard>\d{1,2})\s*\|\s*(?P<subject>.+?)\s*\|\s*(?P<section>.+)$",
    re.IGNORECASE,
)
PAGE_NUMBER = re.compile(r"^\d{1,3}$")
BOOK_PAGE_NUMBER = re.compile(r"^Page\s+\d+$", re.IGNORECASE)
DocumentType = Literal["syllabus", "book_recommendations"]


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    page: int
    standard: str | None
    subject: str | None = None
    page_end: int | None = None


def extract_syllabus_chunks(pdf_path: Path) -> list[DocumentChunk]:
    """Extract overview and grade-specific passages, keeping page/grade metadata."""
    reader = PdfReader(str(pdf_path))
    chunks: list[DocumentChunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_lines = (page.extract_text() or "").splitlines()
        lines = [
            line.strip()
            for line in raw_lines
            if line.strip()
            and not line.strip().startswith("General CBSE-aligned overview |")
            and not PAGE_NUMBER.fullmatch(line.strip())
        ]
        current_heading: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            text = " ".join(current_lines).strip()
            if text:
                chunks.append(DocumentChunk(text, page_number, current_heading))

        for line in lines:
            match = GRADE_HEADING.fullmatch(line)
            if match:
                flush()
                current_heading = f"Standard {int(match.group(1))}"
                current_lines = [current_heading]
            elif (
                line.lower() == "sources and scope"
                or line.startswith("When answering a student or parent,")
                or line.startswith("Common middle-school subject names:")
            ):
                flush()
                current_heading = None
                current_lines = [line]
            elif current_heading is None:
                current_lines.append(line)
            else:
                current_lines.append(line)
        flush()

    if not chunks:
        raise ValueError(f"No readable text was extracted from {pdf_path}.")
    return chunks


def extract_book_recommendation_chunks(pdf_path: Path) -> list[DocumentChunk]:
    """Combine the pages belonging to one Standard-and-subject recommendation record."""
    reader = PdfReader(str(pdf_path))
    records: dict[tuple[str, str], dict[str, object]] = {}

    for page_number, page in enumerate(reader.pages, start=1):
        raw_lines = (page.extract_text() or "").splitlines()
        lines = [
            line.strip()
            for line in raw_lines
            if line.strip()
            and not line.strip().startswith("NCERT-first book recommendations |")
            and not BOOK_PAGE_NUMBER.fullmatch(line.strip())
        ]
        heading_index = next(
            (
                index
                for index, line in enumerate(lines)
                if BOOK_HEADING.fullmatch(line)
            ),
            None,
        )
        heading = (
            BOOK_HEADING.fullmatch(lines[heading_index])
            if heading_index is not None
            else None
        )
        if heading is None:
            # Cover, introduction, and reference pages do not represent a grade-subject record.
            continue

        standard = f"Standard {int(heading.group('standard'))}"
        subject = heading.group("subject").strip()
        section = heading.group("section").strip()
        # PDF text extraction may wrap the final word of a heading onto a
        # separate line (for example, "Recommendation" / "record").
        body_start = heading_index + 1
        if body_start < len(lines) and lines[body_start][:1].islower():
            section = f"{section} {lines[body_start]}"
            body_start += 1
        key = (standard, subject.casefold())
        record = records.setdefault(
            key,
            {"standard": standard, "subject": subject, "page": page_number, "page_end": page_number, "sections": []},
        )
        record["page_end"] = page_number
        body = lines[body_start:]
        record["sections"].append(f"{section}: {' '.join(body)}")

    chunks = [
        DocumentChunk(
            content="\n".join(record["sections"]),
            page=record["page"],
            standard=record["standard"],
            subject=record["subject"],
            page_end=record["page_end"],
        )
        for record in records.values()
        if record["sections"]
    ]
    if not chunks:
        raise ValueError(f"No grade-subject recommendation records were extracted from {pdf_path}.")
    return chunks


def extract_chunks(pdf_path: Path, document_type: DocumentType = "syllabus") -> list[DocumentChunk]:
    """Apply document-aware structure chunking to the selected academic source."""
    if document_type == "syllabus":
        return extract_syllabus_chunks(pdf_path)
    if document_type == "book_recommendations":
        return extract_book_recommendation_chunks(pdf_path)
    raise ValueError(f"Unsupported academic document type: {document_type}")


def chunk_hash(content: str) -> str:
    """Identity of a chunk's embedded text; matches the SQL backfill in migration 005."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ExistingChunk(NamedTuple):
    id: int
    content_hash: str | None
    embedding_model: str


@dataclass(frozen=True)
class ChangePlan:
    kept: list[tuple[int, int]]  # (existing row id, new chunk_index): vector is reused
    inserts: list[int]  # new chunk_index values that need a row
    to_embed: list[str]  # distinct texts that need a new vector
    to_delete: list[int]  # existing row ids that no longer match any chunk


@dataclass(frozen=True)
class IngestResult:
    total: int
    embedded: int
    reused: int
    removed: int


def plan_changes(
    existing: list[ExistingChunk],
    chunks: list[DocumentChunk],
    model: str,
    *,
    force: bool = False,
) -> ChangePlan:
    """Match new chunks to stored rows by (text hash, embedding model).

    Rows are consumed one per chunk, so repeated identical text is matched correctly.
    A row embedded with a different model never matches, so a model change re-embeds all.
    """
    available: dict[str, list[int]] = defaultdict(list)
    if not force:
        for row in existing:
            if row.embedding_model == model and row.content_hash:
                available[row.content_hash.strip()].append(row.id)

    kept: list[tuple[int, int]] = []
    inserts: list[int] = []
    for index, chunk in enumerate(chunks):
        candidates = available.get(chunk_hash(chunk.content))
        if candidates:
            kept.append((candidates.pop(0), index))
        else:
            inserts.append(index)

    kept_ids = {row_id for row_id, _ in kept}
    return ChangePlan(
        kept=kept,
        inserts=inserts,
        to_embed=list(dict.fromkeys(chunks[index].content for index in inserts)),
        to_delete=[row.id for row in existing if row.id not in kept_ids],
    )


def chunk_metadata(
    chunk: DocumentChunk, *, document_type: DocumentType, source_title: str, source_file: str
) -> dict[str, object]:
    return {
        "document_type": document_type,
        "source_title": source_title,
        "source_file": source_file,
        "page": chunk.page,
        "page_end": chunk.page_end or chunk.page,
        "standard": chunk.standard,
        "subject": chunk.subject,
        "academic_year": (
            "general overview; not year-specific"
            if document_type == "syllabus"
            else "2026-2027; verify current school edition"
        ),
        **({"board": "CBSE-aligned"} if document_type == "syllabus" else {}),
    }


# Kept rows are first moved past any final index so renumbering cannot hit the
# UNIQUE (source_id, chunk_index) constraint while rows swap places.
_INDEX_PARK_OFFSET = 1_000_000


async def ingest_chunks(
    chunks: list[DocumentChunk],
    *,
    source_id: str,
    document_type: DocumentType,
    source_title: str,
    source_file: str,
    embedder,
    pool,
    force: bool = False,
) -> IngestResult:
    """Replace a source's chunks, re-embedding only chunks whose text changed.

    Everything happens in one transaction under a per-source advisory lock, so searches see
    either the old or the new set of chunks and concurrent ingests of one source take turns.
    The lock (and one pooled connection) is held while the changed chunks are embedded.
    """

    def metadata_for(chunk: DocumentChunk) -> str:
        return json.dumps(
            chunk_metadata(
                chunk,
                document_type=document_type,
                source_title=source_title,
                source_file=source_file,
            )
        )

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", source_id
            )
            rows = await connection.fetch(
                "SELECT id, content_hash, embedding_model FROM academic_documents "
                "WHERE source_id = $1",
                source_id,
            )
            plan = plan_changes(
                [ExistingChunk(r["id"], r["content_hash"], r["embedding_model"]) for r in rows],
                chunks,
                embedder.model,
                force=force,
            )
            vectors = await embedder.embed_many(plan.to_embed) if plan.to_embed else []
            vector_by_text = dict(zip(plan.to_embed, vectors, strict=True))

            if plan.to_delete:
                await connection.execute(
                    "DELETE FROM academic_documents WHERE id = ANY($1::bigint[])",
                    plan.to_delete,
                )
            if plan.kept:
                await connection.execute(
                    "UPDATE academic_documents SET chunk_index = chunk_index + $2 "
                    "WHERE id = ANY($1::bigint[])",
                    [row_id for row_id, _ in plan.kept],
                    _INDEX_PARK_OFFSET,
                )
                await connection.executemany(
                    "UPDATE academic_documents SET chunk_index = $2, metadata = $3::jsonb "
                    "WHERE id = $1",
                    [
                        (row_id, index, metadata_for(chunks[index]))
                        for row_id, index in plan.kept
                    ],
                )
            for index in plan.inserts:
                chunk = chunks[index]
                await connection.execute(
                    """
                    INSERT INTO academic_documents
                        (source_id, chunk_index, content, metadata, embedding_model,
                         embedding, content_hash)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6::text::vector, $7)
                    """,
                    source_id,
                    index,
                    chunk.content,
                    metadata_for(chunk),
                    embedder.model,
                    to_pgvector(vector_by_text[chunk.content]),
                    chunk_hash(chunk.content),
                )

    return IngestResult(
        total=len(chunks),
        embedded=len(plan.to_embed),
        reused=len(plan.kept),
        removed=len(plan.to_delete),
    )


async def ingest(
    pdf_path: Path, document_type: DocumentType = "syllabus", *, force: bool = False
) -> IngestResult:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Academic PDF not found: {pdf_path}")
    chunks = extract_chunks(pdf_path, document_type)
    # Preserve the original syllabus source ID so a re-ingest replaces legacy
    # syllabus chunks instead of leaving duplicate rows behind.
    source_key = (
        str(pdf_path.resolve())
        if document_type == "syllabus"
        else f"{document_type}:{pdf_path.resolve()}"
    )
    source_title = (
        "Subjects taught in Standards 1 to 10"
        if document_type == "syllabus"
        else "NCERT-first book recommendations for Standards 1 to 10"
    )

    pool = await create_pool()
    try:
        return await ingest_chunks(
            chunks,
            source_id=hashlib.sha256(source_key.encode("utf-8")).hexdigest(),
            document_type=document_type,
            source_title=source_title,
            source_file=pdf_path.name,
            embedder=create_embedder(),
            pool=pool,
            force=force,
        )
    finally:
        await pool.close()


def configured_pdf_path(
    document_type: DocumentType = "syllabus", settings: Settings | None = None
) -> Path:
    """Resolve the selected configured document path relative to the project."""
    configured_settings = settings or get_settings()
    if document_type == "syllabus":
        configured_path = configured_settings.academic_document_path
    elif document_type == "book_recommendations":
        configured_path = configured_settings.academic_book_recommendations_document_path
    else:
        raise ValueError(f"Unsupported academic document type: {document_type}")
    if configured_path.is_absolute():
        return configured_path
    return (PROJECT_ROOT / configured_path).resolve()


async def ingest_configured_pdf(
    document_type: DocumentType = "syllabus", *, force: bool = False
) -> tuple[Path, IngestResult]:
    """Ingest the configured PDF selected by its academic document type."""
    pdf_path = configured_pdf_path(document_type)
    return pdf_path, await ingest(pdf_path, document_type, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the configured academic PDF.")
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="Optional PDF override; otherwise use ACADEMIC_DOCUMENT_PATH from .env.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-embed every chunk instead of only changed ones."
    )
    args = parser.parse_args()
    pdf_path = args.pdf or configured_pdf_path()
    result = asyncio.run(ingest(pdf_path, force=args.force))
    print(
        f"Indexed {result.total} chunks from {pdf_path}: {result.embedded} embedded, "
        f"{result.reused} reused, {result.removed} removed."
    )


if __name__ == "__main__":
    main()
