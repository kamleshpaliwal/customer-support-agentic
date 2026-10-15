"""Load verified fee records from CSV without silently replacing existing data."""

import argparse
import asyncio
import csv
import logging
from decimal import Decimal
from pathlib import Path
from typing import Literal

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, field_validator

from support_agents.db.connection import create_pool


EXPECTED_COLUMNS = [
    "board",
    "standard",
    "academic_year",
    "fee_type",
    "amount",
    "currency",
    "billing_frequency",
]


class FeeSeedRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board: str = Field(min_length=1, max_length=50)
    standard: str = Field(min_length=1, max_length=30)
    academic_year: str = Field(pattern=r"^[0-9]{4}-[0-9]{4}$")
    fee_type: str = Field(min_length=1, max_length=50)
    amount: Decimal = Field(ge=0, max_digits=12, decimal_places=2)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    billing_frequency: Literal["one_time", "monthly", "quarterly", "term", "annual"]

    @field_validator("board", "standard", "fee_type", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("billing_frequency", mode="before")
    @classmethod
    def normalize_frequency(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("academic_year")
    @classmethod
    def require_consecutive_years(cls, value: str) -> str:
        start, end = (int(part) for part in value.split("-"))
        if end != start + 1:
            raise ValueError("academic_year must contain consecutive years, e.g. 2026-2027")
        return value


def read_rows(path: Path) -> list[FeeSeedRow]:
    """Read and validate all rows before connecting to the database."""

    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty; include the required header row.")
        headers = [header.strip() for header in reader.fieldnames]
        if headers != EXPECTED_COLUMNS:
            raise ValueError(f"CSV columns must be exactly: {','.join(EXPECTED_COLUMNS)}")

        rows: list[FeeSeedRow] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        for line_number, raw_row in enumerate(reader, start=2):
            if not raw_row or all(not (value or "").strip() for value in raw_row.values() if value):
                continue
            try:
                row = FeeSeedRow.model_validate(raw_row)
            except Exception as error:
                raise ValueError(f"Invalid CSV row {line_number}: {error}") from error

            key = (
                row.board.casefold(),
                row.standard.casefold(),
                row.academic_year,
                row.fee_type.casefold(),
            )
            if key in seen_keys:
                raise ValueError(f"Duplicate fee component in CSV at row {line_number}.")
            seen_keys.add(key)
            rows.append(row)

    if not rows:
        raise ValueError("CSV contains no fee rows to import.")
    return rows


INSERT_SQL = """
INSERT INTO fees (board, standard, academic_year, fee_type, amount, currency, billing_frequency)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (lower(board), lower(standard), academic_year, lower(fee_type))
DO NOTHING
RETURNING id
"""

UPSERT_SQL = """
INSERT INTO fees (board, standard, academic_year, fee_type, amount, currency, billing_frequency)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (lower(board), lower(standard), academic_year, lower(fee_type))
DO UPDATE SET
    amount = EXCLUDED.amount,
    currency = EXCLUDED.currency,
    billing_frequency = EXCLUDED.billing_frequency
RETURNING id
"""


async def import_rows(rows: list[FeeSeedRow], *, update_existing: bool) -> tuple[int, int]:
    pool = await create_pool()
    inserted = 0
    skipped = 0
    query = UPSERT_SQL if update_existing else INSERT_SQL
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                for row in rows:
                    result = await connection.fetchrow(
                        query,
                        row.board,
                        row.standard,
                        row.academic_year,
                        row.fee_type,
                        row.amount,
                        row.currency,
                        row.billing_frequency,
                    )
                    if result is None:
                        skipped += 1
                    else:
                        inserted += 1
    finally:
        await pool.close()
    return inserted, skipped


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="CSV containing verified fees")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update matching fee rows; by default existing rows are left unchanged.",
    )
    args = parser.parse_args()

    rows = read_rows(args.csv_file)
    inserted, skipped = await import_rows(rows, update_existing=args.update_existing)
    logging.info("Processed %d rows: %d inserted/updated, %d existing rows skipped.", len(rows), inserted, skipped)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
