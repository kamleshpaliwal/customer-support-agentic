"""Regression tests: fee amounts are rendered deterministically from tool evidence."""

from __future__ import annotations

from support_agents.agents.final_responder import _render_verified_fee_answer
from support_agents.schemas import Evidence, SpecialistResult, ToolCallRecord


def _fee_result() -> SpecialistResult:
    return SpecialistResult(
        specialist="finance",
        answer="Fee records were found.",
        evidence=[
            Evidence(source="MCP get_fees", detail="tuition: 52000.00 INR (annual)"),
            Evidence(source="MCP get_fees", detail="learning_resources: 9500.00 INR (annual)"),
            Evidence(source="MCP get_fees", detail="Total: 61500.00 INR (annual)"),
        ],
        tool_calls=[
            ToolCallRecord(
                name="get_fees",
                arguments={
                    "board": "DEMO-CBSE",
                    "standard": "Class 5",
                    "academic_year": "2026-2027",
                },
            )
        ],
    )


def test_total_breakdown_preserves_exact_amounts() -> None:
    answer = _render_verified_fee_answer(
        "How much are all annual fees for DEMO-CBSE Standard 5 in 2026-2027?", _fee_result()
    )
    assert answer is not None
    assert "61,500.00 INR" in answer
    assert "52,000.00 INR" in answer
    assert "9,500.00 INR" in answer
    assert "52,000.0-0" not in answer


def test_single_component_question_reports_only_that_component() -> None:
    answer = _render_verified_fee_answer("What is the tuition fee?", _fee_result())
    assert answer is not None
    assert "52,000.00 INR" in answer
    assert "9,500.00" not in answer
