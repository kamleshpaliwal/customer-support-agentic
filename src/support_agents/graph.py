"""LangGraph workflow construction.

Provider-specific model and MCP implementations are injected as dependencies so the
workflow remains testable and does not silently assume a vendor or database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

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
        request_id: str | None = None,
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
    allow_mcp_elicitation: bool
    user_id: str
    query: str
    intent: Intent
    clarification_question: str | None
    clarification_mode: str | None
    clarification_options: list[str]
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
            request_id=state.get("request_id"),
        )
        update: dict[str, object] = {
            "specialist_result": result,
            "clarification_question": None,
            "clarification_mode": None,
            "clarification_options": [],
        }
        if result.needs_follow_up and result.follow_up_options:
            update.update(
                {
                    "clarification_question": result.answer,
                    "clarification_mode": "fee_board_selection",
                    "clarification_options": result.follow_up_options,
                }
            )
        return update

    async def academic(state: SupportState) -> dict[str, object]:
        result = await dependencies.academic_specialist.respond(
            state["query"],
            user_id=state["user_id"],
            fee_lookup=None,
            request_id=(
                state.get("request_id")
                if state.get("allow_mcp_elicitation", False)
                else None
            ),
        )
        return {"specialist_result": result}

    async def clarification(state: SupportState) -> Command:
        question = state.get("clarification_question")
        options = state.get("clarification_options", [])
        mode = state.get("clarification_mode")
        answer = interrupt(
            {
                "kind": mode or "clarification",
                "question": question
                or "Could you clarify whether your question is about finance or academics?",
                "intent": state.get("intent", Intent.CLARIFICATION).value,
                "options": options,
            }
        )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("A non-empty clarification answer is required to resume the graph.")

        if mode == "fee_board_selection":
            selected_board = next(
                (option for option in options if option.casefold() == answer.strip().casefold()),
                None,
            )
            if selected_board is None and len(options) == 1 and answer.strip().casefold() in {
                "yes",
                "y",
                "correct",
                "use it",
                "use that",
            }:
                selected_board = options[0]

            if selected_board is not None and state.get("fee_lookup") is not None:
                updated_lookup = state["fee_lookup"].model_copy(
                    update={"board": selected_board}
                )
                return Command(
                    goto="finance",
                    update={
                        "fee_lookup": updated_lookup,
                        "clarification_question": None,
                        "clarification_mode": None,
                        "clarification_options": [],
                    },
                )

            result = SpecialistResult(
                specialist="finance",
                answer=(
                    "Understood. I cancelled the lookup because no exact board was confirmed. "
                    "Please submit a new question with the board name you want checked."
                ),
                limitations=["No fee record was queried for a similar board without confirmation."],
            )
            return Command(
                goto="final_response",
                update={
                    "specialist_result": result,
                    "clarification_question": None,
                    "clarification_mode": None,
                    "clarification_options": [],
                },
            )

        # Re-run routing against the original question plus the user's clarification.
        # The specialist will only be called after the required fields validate.
        combined_query = (
            f"Original request: {state['query']}\n"
            f"User clarification: {answer.strip()}"
        )
        return Command(
            goto="detect_intent",
            update={"query": combined_query, "clarification_question": None},
        )

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

    def route_after_finance(state: SupportState) -> str:
        return "clarification" if state.get("clarification_question") else "final_response"

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
    builder.add_conditional_edges(
        "finance",
        route_after_finance,
        {"clarification": "clarification", "final_response": "final_response"},
    )
    for node in ("academic", "unsupported"):
        builder.add_edge(node, "final_response")
    builder.add_edge("final_response", END)
    # Keeps paused conversations resumable across HTTP requests while this process runs.
    # Replace with a durable checkpointer (for example PostgreSQL) for multi-worker production.
    return builder.compile(checkpointer=InMemorySaver())
