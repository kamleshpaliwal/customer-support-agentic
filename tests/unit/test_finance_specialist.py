"""Unit tests for fee handling at the finance specialist's MCP boundary."""

from __future__ import annotations

import asyncio
from typing import cast

from support_agents.agents.finance_specialist import FinanceSpecialist
from support_agents.mcp.client import FeeMCPClientError, FeesMCPClient
from support_agents.schemas import FeeItem, FeeLookupRequest, FeeLookupResult


class FakeFeesClient:
    def __init__(self, *responses: FeeLookupResult | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[FeeLookupRequest] = []

    async def get_fees(self, request: FeeLookupRequest) -> FeeLookupResult:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def found_result() -> FeeLookupResult:
    return FeeLookupResult(
        found=True,
        board="DEMO-CBSE",
        standard="Class 1",
        academic_year="2026-2027",
        fee_items=[
            FeeItem(
                fee_type="tuition",
                amount="42000.00",
                currency="INR",
                billing_frequency="annual",
            ),
            FeeItem(
                fee_type="learning_resources",
                amount="8000.00",
                currency="INR",
                billing_frequency="annual",
            ),
        ],
        total_amount="50000.00",
        currency="INR",
        billing_frequency="annual",
        message="Fee record found.",
    )


def no_match_result(*, candidates: list[str] | None = None) -> FeeLookupResult:
    return FeeLookupResult(
        found=False,
        board="CBSE",
        standard="Class 1",
        academic_year="2026-2027",
        fee_items=[],
        total_amount=None,
        currency=None,
        billing_frequency=None,
        message="No fee record was found for the requested board, standard, and academic year.",
        board_candidates=candidates or [],
    )


def fee_request(board: str = "DEMO-CBSE") -> FeeLookupRequest:
    return FeeLookupRequest(
        board=board,
        standard="Standard 1",
        academic_year="2026-2027",
    )


def finance_specialist(client: FakeFeesClient) -> FinanceSpecialist:
    return FinanceSpecialist(fees_client=cast(FeesMCPClient, client))


def test_successful_lookup_returns_fee_evidence_and_records_tool_arguments() -> None:
    client = FakeFeesClient(found_result())
    specialist = finance_specialist(client)
    request = fee_request()

    result = asyncio.run(
        specialist.respond(
            "What is the fee?",
            user_id="test-user",
            fee_lookup=request,
        )
    )

    assert client.requests == [request]
    assert result.specialist == "finance"
    assert not result.needs_follow_up
    assert result.limitations == []
    assert [evidence.detail for evidence in result.evidence] == [
        "tuition: 42000.00 INR (annual)",
        "learning_resources: 8000.00 INR (annual)",
        "Total: 50000.00 INR (annual)",
    ]
    assert result.tool_calls[0].name == "get_fees"
    assert result.tool_calls[0].arguments == {
        "board": "DEMO-CBSE",
        "standard": "Class 1",
        "academic_year": "2026-2027",
    }


def test_missing_lookup_does_not_call_mcp_and_requests_required_fields() -> None:
    client = FakeFeesClient()
    specialist = finance_specialist(client)

    result = asyncio.run(specialist.respond("How much are the fees?", user_id="test-user"))

    assert client.requests == []
    assert result.needs_follow_up
    assert "board, standard, and academic year" in result.answer


def test_no_match_returns_limitation_without_claiming_a_fee_was_found() -> None:
    client = FakeFeesClient(no_match_result())
    specialist = finance_specialist(client)

    result = asyncio.run(
        specialist.respond(
            "What is the fee?",
            user_id="test-user",
            fee_lookup=fee_request("CBSE"),
        )
    )

    assert len(client.requests) == 1
    assert result.evidence == []
    assert result.limitations == ["No fee record was found for the requested board, standard, and academic year."]
    assert not result.needs_follow_up


def test_exact_board_candidate_in_user_query_allows_one_confirmed_retry() -> None:
    client = FakeFeesClient(
        no_match_result(candidates=["DEMO-CBSE"]),
        found_result(),
    )
    specialist = finance_specialist(client)

    result = asyncio.run(
        specialist.respond(
            "What is the tuition fee for DEMO-CBSE?",
            user_id="test-user",
            fee_lookup=fee_request("CBSE"),
        )
    )

    assert [request.board for request in client.requests] == ["CBSE", "DEMO-CBSE"]
    assert not result.needs_follow_up
    assert len(result.evidence) == 3
    assert [call.arguments["board"] for call in result.tool_calls] == ["CBSE", "DEMO-CBSE"]


def test_unconfirmed_board_candidate_prompts_user_without_retrying() -> None:
    client = FakeFeesClient(no_match_result(candidates=["DEMO-CBSE"]))
    specialist = finance_specialist(client)

    result = asyncio.run(
        specialist.respond(
            "What is the tuition fee for CBSE?",
            user_id="test-user",
            fee_lookup=fee_request("CBSE"),
        )
    )

    assert len(client.requests) == 1
    assert result.needs_follow_up
    assert result.follow_up_options == ["DEMO-CBSE"]
    assert "Which board should I use?" in result.answer


def test_mcp_client_error_returns_safe_fallback() -> None:
    client = FakeFeesClient(FeeMCPClientError("MCP unavailable"))
    specialist = finance_specialist(client)

    result = asyncio.run(
        specialist.respond(
            "What is the fee?",
            user_id="test-user",
            fee_lookup=fee_request(),
        )
    )

    assert result.answer == "I could not retrieve the fee records right now."
    assert result.limitations == ["MCP unavailable"]
    assert result.tool_calls[0].name == "get_fees"
