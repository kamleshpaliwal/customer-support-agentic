"""Typed contracts shared by the graph, agents, and MCP adapters."""

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Intent(StrEnum):
    FINANCE = "finance"
    ACADEMIC = "academic"
    CLARIFICATION = "clarification"
    UNSUPPORTED = "unsupported"


class FeeLookupRequest(BaseModel):
    board: str = Field(min_length=1, max_length=50)
    standard: str = Field(min_length=1, max_length=30)
    academic_year: str = Field(pattern=r"^[0-9]{4}-[0-9]{4}$")

    @field_validator("board", "standard", mode="before")
    @classmethod
    def strip_lookup_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("standard")
    @classmethod
    def normalize_numeric_standard(cls, value: str) -> str:
        """Normalize `5`, `Standard 5`, and `Grade 5` to the seeded `Class 5` form."""
        match = re.fullmatch(
            r"(?:(?:class|standard|grade)\s*)?(\d{1,2})", value, re.IGNORECASE
        )
        if match:
            return f"Class {int(match.group(1))}"
        return value

    @field_validator("academic_year")
    @classmethod
    def require_consecutive_years(cls, value: str) -> str:
        start, end = (int(part) for part in value.split("-"))
        if end != start + 1:
            raise ValueError("Academic year must contain consecutive years, e.g. 2026-2027")
        return value


class IntentDecision(BaseModel):
    intent: Intent
    rationale: str = Field(description="Brief routing rationale; do not include hidden chain of thought.")
    clarification_question: str | None = None
    fee_lookup: FeeLookupRequest | None = None


class Evidence(BaseModel):
    source: str
    detail: str


class FeeItem(BaseModel):
    fee_type: str
    amount: str = Field(description="Exact monetary amount as a decimal string.")
    currency: str
    billing_frequency: str


class FeeLookupResult(BaseModel):
    found: bool
    board: str
    standard: str
    academic_year: str
    fee_items: list[FeeItem]
    total_amount: str | None = Field(
        description="Exact sum when all components share currency and billing frequency; otherwise null."
    )
    currency: str | None
    billing_frequency: str | None
    message: str
    board_candidates: list[str] = Field(default_factory=list)


class AcademicSearchMatch(BaseModel):
    content: str
    source: str
    page: int | None = None
    standard: str | None = None
    subject: str | None = None
    similarity_score: float = Field(ge=-1.0, le=1.0)


class AcademicSearchResult(BaseModel):
    matches: list[AcademicSearchMatch] = Field(default_factory=list)
    message: str
    language: Literal["English", "Hindi"] | None = None
    elicitation_cancelled: bool = False


class SpecialistResult(BaseModel):
    specialist: str
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    needs_follow_up: bool = False
    follow_up_options: list[str] = Field(default_factory=list)


class SupportRequest(BaseModel):
    request_id: str
    user_id: str
    query: str = Field(min_length=1, max_length=20_000)


class SupportResponse(BaseModel):
    request_id: str
    answer: str
    intent: Intent
    limitations: list[str] = Field(default_factory=list)
