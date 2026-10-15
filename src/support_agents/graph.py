"""LangGraph workflow construction.

Provider-specific model and MCP implementations are injected as dependencies so the
workflow remains testable and does not silently assume a vendor or database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from support_agents.schemas import FeeLookupRequest, Intent, IntentDecision, SpecialistResult


class IntentDetector(Protocol):
    async def detect(self, query: str) -> IntentDecision: ...


class Specialist(Protocol):
    async def respond(
        self,
        query: str,
        *,
        user_id: str,
        fee_lookup: FeeLookupRequest | None = None,
    ) -> SpecialistResult: ...


class FinalResponder(Protocol):
    async def render(self, query: str, result: SpecialistResult) -> str: ...


@dataclass(frozen=True)
class GraphDependencies:
    intent_detector: IntentDetector
    finance_specialist: Specialist
    academic_specialist: Specialist
    final_responder: FinalResponder


class SupportState(TypedDict, total=False):
    request_id: str
    user_id: str
    query: str
    intent: Intent
    clarification_question: str | None
    fee_lookup: FeeLookupRequest | None
    specialist_result: SpecialistResult
    answer: str
    limitations: list[str]


def build_support_graph(dependencies: GraphDependencies):
    """Build and compile the support request graph with injected runtime adapters."""

    async def detect_intent(state: SupportState) -> dict[str, object]:
        decision = await dependencies.intent_detector.detect(state["query"])
        return {
            "intent": decision.intent,
            "clarification_question": decision.clarification_question,
            "fee_lookup": decision.fee_lookup,
        }

    async def finance(state: SupportState) -> dict[str, object]:
        result = await dependencies.finance_specialist.respond(
            state["query"],
            user_id=state["user_id"],
            fee_lookup=state.get("fee_lookup"),
        )
        return {"specialist_result": result}

    async def academic(state: SupportState) -> dict[str, object]:
        result = await dependencies.academic_specialist.respond(
            state["query"], user_id=state["user_id"], fee_lookup=None
        )
        return {"specialist_result": result}

    async def clarification(state: SupportState) -> dict[str, object]:
        question = state.get("clarification_question")
        result = SpecialistResult(
            specialist="supervisor",
            answer=question or "Could you clarify whether your question is about finance or academics?",
            needs_follow_up=True,
        )
        return {"specialist_result": result}

    async def unsupported(_: SupportState) -> dict[str, object]:
        result = SpecialistResult(
            specialist="supervisor",
            answer="I can help with finance or academic questions. Please rephrase your request in one of those areas.",
            limitations=["The request did not match a supported area."],
        )
        return {"specialist_result": result}

    async def final_response(state: SupportState) -> dict[str, object]:
        result = state["specialist_result"]
        answer = await dependencies.final_responder.render(state["query"], result)
        return {"answer": answer, "limitations": result.limitations}

    def route(state: SupportState) -> str:
        return {
            Intent.FINANCE: "finance",
            Intent.ACADEMIC: "academic",
            Intent.CLARIFICATION: "clarification",
            Intent.UNSUPPORTED: "unsupported",
        }[state["intent"]]

    builder = StateGraph(SupportState)
    builder.add_node("detect_intent", detect_intent)
    builder.add_node("finance", finance)
    builder.add_node("academic", academic)
    builder.add_node("clarification", clarification)
    builder.add_node("unsupported", unsupported)
    builder.add_node("final_response", final_response)
    builder.add_edge(START, "detect_intent")
    builder.add_conditional_edges(
        "detect_intent",
        route,
        {
            "finance": "finance",
            "academic": "academic",
            "clarification": "clarification",
            "unsupported": "unsupported",
        },
    )
    for node in ("finance", "academic", "clarification", "unsupported"):
        builder.add_edge(node, "final_response")
    builder.add_edge("final_response", END)
    return builder.compile()
