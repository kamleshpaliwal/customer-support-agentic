"""Unit tests for the LangGraph workflow using deterministic local fakes."""

from __future__ import annotations

import asyncio

import pytest

from support_agents.graph import GraphDependencies, build_support_graph
from support_agents.schemas import Intent, IntentDecision, SpecialistResult


class FakeIntentDetector:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent

    async def detect(self, query: str) -> IntentDecision:
        return IntentDecision(
            intent=self.intent,
            rationale="Deterministic decision for graph unit testing.",
            clarification_question=(
                "Could you clarify your question?"
                if self.intent == Intent.CLARIFICATION
                else None
            ),
        )


class FakeSpecialist:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, str]] = []

    async def respond(
        self,
        query: str,
        *,
        user_id: str,
        fee_lookup: object = None,
        request_id: str | None = None,
    ) -> SpecialistResult:
        del fee_lookup, request_id
        self.calls.append((query, user_id))
        return SpecialistResult(
            specialist=self.name,
            answer=f"{self.name} result for: {query}",
        )


class FakeFinalResponder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SpecialistResult]] = []

    async def render(self, query: str, result: SpecialistResult) -> str:
        self.calls.append((query, result))
        return f"Final answer based on {result.specialist}: {result.answer}"


@pytest.mark.parametrize(
    ("intent", "expected_specialist"),
    [
        (Intent.FINANCE, "finance"),
        (Intent.ACADEMIC, "academic"),
        (Intent.UNSUPPORTED, "supervisor"),
    ],
)
def test_graph_routes_to_expected_specialist_and_final_responder(
    intent: Intent,
    expected_specialist: str,
) -> None:
    detector = FakeIntentDetector(intent)
    finance = FakeSpecialist("finance")
    academic = FakeSpecialist("academic")
    final_responder = FakeFinalResponder()
    graph = build_support_graph(
        GraphDependencies(
            intent_detector=detector,
            finance_specialist=finance,
            academic_specialist=academic,
            final_responder=final_responder,
        )
    )
    query = "Test customer support query"

    result = asyncio.run(
        graph.ainvoke(
            {"query": query, "user_id": "test-user"},
            config={"configurable": {"thread_id": f"route-{intent.value}"}},
        )
    )

    assert result["intent"] == intent
    assert result["specialist_result"].specialist == expected_specialist
    assert result["answer"].startswith(f"Final answer based on {expected_specialist}:")
    assert len(final_responder.calls) == 1
    assert final_responder.calls[0][0] == query

    if intent == Intent.FINANCE:
        assert finance.calls == [(query, "test-user")]
        assert academic.calls == []
    elif intent == Intent.ACADEMIC:
        assert academic.calls == [(query, "test-user")]
        assert finance.calls == []
    else:
        assert finance.calls == []
        assert academic.calls == []


def test_clarification_intent_interrupts_for_user_input() -> None:
    detector = FakeIntentDetector(Intent.CLARIFICATION)
    finance = FakeSpecialist("finance")
    academic = FakeSpecialist("academic")
    final_responder = FakeFinalResponder()
    graph = build_support_graph(
        GraphDependencies(
            intent_detector=detector,
            finance_specialist=finance,
            academic_specialist=academic,
            final_responder=final_responder,
        )
    )

    result = asyncio.run(
        graph.ainvoke(
            {"query": "I need help", "user_id": "test-user"},
            config={"configurable": {"thread_id": "clarification-test"}},
        )
    )

    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    assert interrupts[0].value["kind"] == "clarification"
    assert interrupts[0].value["question"] == "Could you clarify your question?"
    assert finance.calls == []
    assert academic.calls == []
    assert final_responder.calls == []
