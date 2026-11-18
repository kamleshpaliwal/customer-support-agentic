"""Listing fees by billing frequency: routing, board question, rendering and the full graph."""

from __future__ import annotations

import asyncio

import pytest
from langgraph.types import Command

from support_agents.agents.final_responder import _render_fee_listing
from support_agents.agents.finance_specialist import FinanceSpecialist
from support_agents.agents.supervisor import _RoutingExtraction, _to_intent_decision
from support_agents.graph import GraphDependencies, build_support_graph
from support_agents.mcp.client import FeeMCPClientError
from support_agents.schemas import (
    ALL_BOARDS,
    FeeListingRequest,
    FeeListItem,
    FeeListResult,
    Intent,
    IntentDecision,
    SpecialistResult,
)


def item(board, standard, fee_type, amount, year="2026-2027") -> FeeListItem:
    return FeeListItem(
        board=board, standard=standard, academic_year=year, fee_type=fee_type,
        amount=amount, currency="INR",
    )


ROWS = [
    item("DEMO-CBSE", "Class 1", "learning_resources", "8000.00"),
    item("DEMO-CBSE", "Class 1", "tuition", "42000.00"),
    item("DEMO-ICSE", "Class 1", "tuition", "46000.00"),
]


class FakeFeesClient:
    """Mimics the MCP tool: filters rows, reports boards, and counts calls."""

    def __init__(self, rows=ROWS, fail=False, truncate=False):
        self.rows, self.fail, self.truncate, self.requests = rows, fail, truncate, []

    async def list_fees(self, request: FeeListingRequest) -> FeeListResult:
        self.requests.append(request)
        if self.fail:
            raise FeeMCPClientError("boom")
        rows = [r for r in self.rows if request.board is None or r.board == request.board]
        return FeeListResult(
            billing_frequency=request.billing_frequency,
            items=rows,
            total_matches=len(rows) + (10 if self.truncate else 0),
            truncated=self.truncate,
            boards=sorted({r.board for r in rows}),
            message=f"Found {len(rows)} records.",
        )


def extraction(**overrides) -> _RoutingExtraction:
    values = dict(intent="finance", board="", standard="", academic_year="",
                  billing_frequency="annual", clarification_question="")
    return _RoutingExtraction(**{**values, **overrides})


# ---------- routing

def test_a_frequency_question_needs_no_board_class_or_year():
    decision = _to_intent_decision(extraction())
    assert decision.intent is Intent.FINANCE
    assert decision.fee_listing == FeeListingRequest(billing_frequency="annual")
    assert decision.fee_lookup is None and decision.clarification_question is None


def test_filters_that_are_stated_are_kept():
    decision = _to_intent_decision(extraction(board="DEMO-CBSE", standard="5"))
    assert decision.fee_listing.board == "DEMO-CBSE"
    assert decision.fee_listing.standard == "Class 5"


def test_board_class_and_year_together_stay_a_specific_lookup():
    decision = _to_intent_decision(
        extraction(board="DEMO-CBSE", standard="5", academic_year="2026-2027")
    )
    assert decision.fee_lookup is not None and decision.fee_listing is None


def test_an_invalid_standard_is_clarified_not_guessed():
    decision = _to_intent_decision(extraction(standard="11"))
    assert decision.intent is Intent.CLARIFICATION
    assert "standard" in decision.clarification_question


def test_no_frequency_keeps_the_old_behaviour():
    decision = _to_intent_decision(extraction(billing_frequency=""))
    assert decision.intent is Intent.CLARIFICATION  # still asks for board, class and year


def test_a_missing_frequency_field_from_a_small_model_defaults_to_empty():
    model = _RoutingExtraction.model_validate(
        {"intent": "finance", "board": "", "standard": "", "academic_year": "",
         "clarification_question": ""}
    )
    assert model.billing_frequency == ""


# ---------- finance specialist

def respond(client, listing):
    return asyncio.run(FinanceSpecialist(client).respond("q", user_id="u", fee_listing=listing))


def test_several_boards_ask_one_question_with_options():
    result = respond(FakeFeesClient(), FeeListingRequest(billing_frequency="annual"))
    assert result.needs_follow_up
    assert result.answer == "Which board's annual fees should I list?"
    assert result.follow_up_options == ["DEMO-CBSE", "DEMO-ICSE", ALL_BOARDS]


def test_a_single_board_is_listed_without_asking():
    client = FakeFeesClient(rows=ROWS[:2])
    result = respond(client, FeeListingRequest(billing_frequency="annual"))
    assert not result.needs_follow_up and len(result.evidence) == 2


def test_a_named_board_is_listed_without_asking():
    result = respond(FakeFeesClient(), FeeListingRequest(billing_frequency="annual", board="DEMO-ICSE"))
    assert not result.needs_follow_up and len(result.evidence) == 1


def test_all_boards_is_listed_without_asking_again():
    result = respond(FakeFeesClient(), FeeListingRequest(billing_frequency="annual", all_boards=True))
    assert not result.needs_follow_up and len(result.evidence) == 3


def test_no_matching_fees_is_a_plain_message():
    result = respond(FakeFeesClient(rows=[]), FeeListingRequest(billing_frequency="monthly"))
    assert not result.needs_follow_up and not result.evidence


def test_a_truncated_list_says_how_many_were_not_shown():
    result = respond(
        FakeFeesClient(truncate=True), FeeListingRequest(billing_frequency="annual", all_boards=True)
    )
    assert "Showing 3 of 13 annual fee records" in result.answer
    assert result.limitations


def test_a_tool_failure_is_reported_gracefully():
    result = respond(FakeFeesClient(fail=True), FeeListingRequest(billing_frequency="annual"))
    assert "could not retrieve" in result.answer


# ---------- rendering

def test_rendering_groups_by_board_and_class_with_exact_amounts():
    result = respond(FakeFeesClient(), FeeListingRequest(billing_frequency="annual", all_boards=True))
    text = _render_fee_listing(result)
    assert text.splitlines() == [
        "Found 3 annual fee records.",
        "",
        "DEMO-CBSE, 2026-2027",
        "- Class 1: learning resources 8,000.00 INR; tuition 42,000.00 INR",
        "",
        "DEMO-ICSE, 2026-2027",
        "- Class 1: tuition 46,000.00 INR",
    ]


def test_an_empty_list_is_stated_plainly_for_any_responder():
    from support_agents.agents.final_responder import _render_verified_fee_answer

    result = respond(FakeFeesClient(rows=[]), FeeListingRequest(billing_frequency="monthly"))
    assert _render_verified_fee_answer("q", result) == "Found 0 records."


def test_other_answers_are_not_rendered_as_a_listing():
    assert _render_fee_listing(SpecialistResult(specialist="finance", answer="x")) is None


# ---------- the whole graph

class Detector:
    async def detect(self, query):
        return IntentDecision(
            intent=Intent.FINANCE, rationale="x",
            fee_listing=FeeListingRequest(billing_frequency="annual"),
        )


class Academic:
    async def respond(self, *a, **k):  # pragma: no cover - never routed here
        raise AssertionError


class Responder:
    async def render(self, query, result):
        return _render_fee_listing(result) or result.answer


def run_graph(thread: str):
    client = FakeFeesClient()
    graph = build_support_graph(
        GraphDependencies(
            intent_detector=Detector(), finance_specialist=FinanceSpecialist(client),
            academic_specialist=Academic(), final_responder=Responder(),
        )
    )
    config = {"configurable": {"thread_id": thread}}
    first = asyncio.run(graph.ainvoke({"query": "What fees are charged annually?", "user_id": "u"}, config=config))
    return graph, config, first


def resume(graph, config, answer):
    return asyncio.run(graph.ainvoke(Command(resume=answer), config=config))


def test_graph_asks_for_a_board_with_options():
    _, _, first = run_graph("listing-ask")
    payload = first["__interrupt__"][0].value
    assert payload["kind"] == "fee_listing_board_selection"
    assert payload["options"] == ["DEMO-CBSE", "DEMO-ICSE", ALL_BOARDS]


def test_choosing_a_board_lists_only_that_board():
    graph, config, _ = run_graph("listing-board")
    answer = resume(graph, config, "DEMO-ICSE")["answer"]
    assert "DEMO-ICSE" in answer and "DEMO-CBSE" not in answer


def test_choosing_all_boards_lists_everything():
    graph, config, _ = run_graph("listing-all")
    answer = resume(graph, config, ALL_BOARDS)["answer"]
    assert "DEMO-ICSE" in answer and "DEMO-CBSE" in answer


@pytest.mark.parametrize("reply", ["no", "something else"])
def test_any_other_answer_cancels_the_list(reply):
    graph, config, _ = run_graph(f"listing-cancel-{reply}")
    answer = resume(graph, config, reply)["answer"]
    assert "cancelled" in answer
