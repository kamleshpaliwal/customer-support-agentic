"""Unit tests for request schema validation and normalization."""

import pytest
from pydantic import ValidationError

from support_agents.schemas import FeeLookupRequest


@pytest.mark.parametrize(
    ("standard", "expected"),
    [
        ("Standard 5", "Class 5"),
        ("Grade 5", "Class 5"),
        ("5", "Class 5"),
        (" class 10 ", "Class 10"),
    ],
)
def test_fee_lookup_normalizes_supported_standard(standard: str, expected: str) -> None:
    request = FeeLookupRequest(
        board="DEMO-CBSE",
        standard=standard,
        academic_year="2026-2027",
    )

    assert request.standard == expected


def test_fee_lookup_strips_surrounding_whitespace() -> None:
    request = FeeLookupRequest(
        board="  DEMO-CBSE  ",
        standard="  Grade 5  ",
        academic_year="2026-2027",
    )

    assert request.board == "DEMO-CBSE"
    assert request.standard == "Class 5"


@pytest.mark.parametrize("standard", ["Class 0", "Class 11", "Kindergarten", ""])
def test_fee_lookup_rejects_unsupported_standard(standard: str) -> None:
    with pytest.raises(ValidationError):
        FeeLookupRequest(
            board="DEMO-CBSE",
            standard=standard,
            academic_year="2026-2027",
        )


@pytest.mark.parametrize("academic_year", ["2026", "2026/2027", "2026-2028"])
def test_fee_lookup_rejects_invalid_academic_year(academic_year: str) -> None:
    with pytest.raises(ValidationError):
        FeeLookupRequest(
            board="DEMO-CBSE",
            standard="Class 5",
            academic_year=academic_year,
        )
