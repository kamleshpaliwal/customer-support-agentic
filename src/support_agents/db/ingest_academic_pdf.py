"""Ingest configured academic PDFs into pgvector.

Run with: python -m support_agents.db.ingest_academic_pdf [path/to/document.pdf]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from support_agents.db.connection import create_pool
from support_agents.embeddings import OllamaEmbedder, to_pgvector
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


async def ingest(pdf_path: Path, document_type: DocumentType = "syllabus") -> int:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Academic PDF not found: {pdf_path}")
    chunks = extract_chunks(pdf_path, document_type)
    embedder = OllamaEmbedder()
    vectors = await embedder.embed_many([chunk.content for chunk in chunks])
    # Preserve the original syllabus source ID so a re-ingest replaces legacy
    # syllabus chunks instead of leaving duplicate rows behind.
    source_key = (
        str(pdf_path.resolve())
        if document_type == "syllabus"
        else f"{document_type}:{pdf_path.resolve()}"
    )
    source_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
    source_title = (
        "Subjects taught in Standards 1 to 10"
        if document_type == "syllabus"
        else "NCERT-first book recommendations for Standards 1 to 10"
    )

    pool = await create_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    source_id,
                )
                await connection.execute(
                    "DELETE FROM academic_documents WHERE source_id = $1", source_id
                )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                    metadata = {
                        "document_type": document_type,
                        "source_title": source_title,
                        "source_file": pdf_path.name,
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
                    await connection.execute(
                        """
                        INSERT INTO academic_documents
                            (source_id, chunk_index, content, metadata, embedding_model, embedding)
                        VALUES ($1, $2, $3, $4::jsonb, $5, $6::text::vector)
                        """,
                        source_id,
                        index,
                        chunk.content,
                        json.dumps(metadata),
                        embedder.model,
                        to_pgvector(vector),
                    )
    finally:
        await pool.close()
    return len(chunks)


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
    document_type: DocumentType = "syllabus",
) -> tuple[Path, int]:
    """Ingest the configured PDF selected by its academic document type."""
    pdf_path = configured_pdf_path(document_type)
    return pdf_path, await ingest(pdf_path, document_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the configured academic PDF.")
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="Optional PDF override; otherwise use ACADEMIC_DOCUMENT_PATH from .env.",
    )
    args = parser.parse_args()
    pdf_path = args.pdf or configured_pdf_path()
    count = asyncio.run(ingest(pdf_path))
    print(f"Indexed {count} chunks from {pdf_path}.")


if __name__ == "__main__":
    main()
