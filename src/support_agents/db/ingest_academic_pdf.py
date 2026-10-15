"""Ingest the approved academic overview PDF into pgvector.

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

from pypdf import PdfReader

from support_agents.db.connection import create_pool
from support_agents.embeddings import OllamaEmbedder, to_pgvector
from support_agents.settings import PROJECT_ROOT, Settings, get_settings


GRADE_HEADING = re.compile(r"^Standard\s+(\d{1,2})$", re.IGNORECASE)
PAGE_NUMBER = re.compile(r"^\d{1,3}$")


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    page: int
    standard: str | None


def extract_chunks(pdf_path: Path) -> list[DocumentChunk]:
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


async def ingest(pdf_path: Path) -> int:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Academic PDF not found: {pdf_path}")
    chunks = extract_chunks(pdf_path)
    embedder = OllamaEmbedder()
    vectors = await embedder.embed_many([chunk.content for chunk in chunks])
    source_id = hashlib.sha256(str(pdf_path.resolve()).encode("utf-8")).hexdigest()
    source_title = "Subjects taught in Standards 1 to 10"

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
                        "source_title": source_title,
                        "source_file": pdf_path.name,
                        "page": chunk.page,
                        "standard": chunk.standard,
                        "board": "CBSE-aligned",
                        "academic_year": "general overview; not year-specific",
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


def configured_pdf_path(settings: Settings | None = None) -> Path:
    """Resolve the configured document path relative to the project when needed."""
    configured_path = (settings or get_settings()).academic_document_path
    if configured_path.is_absolute():
        return configured_path
    return (PROJECT_ROOT / configured_path).resolve()


async def ingest_configured_pdf() -> tuple[Path, int]:
    """Ingest the PDF selected by ACADEMIC_DOCUMENT_PATH."""
    pdf_path = configured_pdf_path()
    return pdf_path, await ingest(pdf_path)


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
