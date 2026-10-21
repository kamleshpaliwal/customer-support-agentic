"""Run the labeled support dataset against the configured local support graph."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from support_agents.agents.academic_specialist import AcademicSpecialist
from support_agents.agents.final_responder import create_final_responder
from support_agents.agents.finance_specialist import FinanceSpecialist
from support_agents.agents.supervisor import create_intent_detector
from support_agents.graph import GraphDependencies, build_support_graph
from support_agents.mcp.elicitation import elicitation_broker
from support_agents.settings import PROJECT_ROOT, configure_langsmith_tracing, get_settings

DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "support_eval.jsonl"


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    category: str
    user_input: str
    expected_intent: str
    expected_action: str
    expected_tool: str | None = None
    expected_tool_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_fee: dict[str, str] | None = None
    reference: str
    reference_contexts: list[str] = Field(default_factory=list)


def load_dataset(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"Invalid evaluation case at {path}:{line_number}: {error}") from error
    return cases


def _make_graph():
    settings = get_settings()
    configure_langsmith_tracing(settings)
    return build_support_graph(
        GraphDependencies(
            intent_detector=create_intent_detector(settings),
            finance_specialist=FinanceSpecialist(),
            academic_specialist=AcademicSpecialist(),
            final_responder=create_final_responder(settings),
        )
    )


async def _invoke_case(graph, case: EvaluationCase, timeout_seconds: int) -> dict[str, Any]:
    request_id = str(uuid4())
    started_at = time.perf_counter()
    expects_elicitation = case.expected_tool == "search_book_recommendations"
    input_state = {
        "request_id": request_id,
        "user_id": f"eval-{case.id}",
        "query": case.user_input,
        "allow_mcp_elicitation": expects_elicitation,
    }
    invocation = asyncio.create_task(
        graph.ainvoke(
            input_state,
            config={"configurable": {"thread_id": request_id}},
        )
    )
    elicitation: dict[str, Any] | None = None
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    try:
        while not invocation.done():
            pending = elicitation_broker.get_pending(request_id) if expects_elicitation else None
            if pending is not None:
                expected_clarification = case.model_extra.get("expected_clarification") or {}
                is_expected_user_cancel = expected_clarification.get("user_action") == "cancel"
                elicitation = {
                    "question": pending.message,
                    "field": pending.field_name,
                    "options": pending.options,
                    "resolution": (
                        "cancelled_by_dataset"
                        if is_expected_user_cancel
                        else "cancelled_after_observation"
                    ),
                }
                # The dataset checks whether the system asks. Cancel after observing the prompt,
                # so the offline evaluator never silently chooses a language for the user.
                elicitation_broker.submit(request_id, "cancel")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Case exceeded the {timeout_seconds}s evaluation timeout")
            await asyncio.sleep(0.05)

        state = await invocation
    except BaseException:
        if not invocation.done():
            invocation.cancel()
            await asyncio.gather(invocation, return_exceptions=True)
        raise

    specialist = state.get("specialist_result")
    if hasattr(specialist, "model_dump"):
        specialist = specialist.model_dump(mode="json")
    specialist = specialist or {}
    interrupt_values = state.get("__interrupt__", ())
    interrupt_value = interrupt_values[0].value if interrupt_values else None
    intent = state.get("intent")
    if hasattr(intent, "value"):
        intent = intent.value

    return {
        "request_id": request_id,
        "intent": intent,
        "answer": state.get("answer"),
        "limitations": state.get("limitations", []),
        "specialist": specialist,
        "clarification": interrupt_value,
        "mcp_elicitation": elicitation,
        "node_timings_ms": state.get("node_timings_ms", {}),
        "duration_seconds": round(time.perf_counter() - started_at, 3),
    }


def _tool_argument_matches(expected: dict[str, Any], calls: list[dict[str, Any]]) -> bool | None:
    if not expected:
        return None
    if not calls:
        return False

    # A lookup can retry with a fuller exact candidate (for example, CBSE -> DEMO-CBSE).
    # Score the final invocation because that is the one whose result feeds the answer.
    name = calls[-1].get("name")
    actual = calls[-1].get("arguments", {})
    for key, expected_value in expected.items():
        if key == "language" and expected_value == "ask_user_to_choose":
            continue  # Assessed using the actual MCP elicitation event below.
        if key == "language" and expected_value == "cancelled":
            continue  # Assessed from the recorded cancellation event.
        actual_value = actual.get(key)
        if actual_value is None and key in {
            "standard",
            "subject",
            "language",
            "academic_year",
        }:
            actual_value = actual.get("query", "")
        if actual_value is None or str(expected_value).casefold() not in str(actual_value).casefold():
            return False
        if name == "get_fees" and str(expected_value).casefold() != str(actual_value).casefold():
            return False
    return True


def _observed_action(actual: dict[str, Any]) -> str:
    if actual.get("mcp_elicitation"):
        if actual["mcp_elicitation"].get("resolution") == "cancelled_by_dataset":
            return "cancel_without_search_or_title"
        return "elicit_language"
    clarification = actual.get("clarification")
    if clarification:
        if clarification.get("options"):
            return "confirm_board_candidate_before_lookup"
        return "ask_for_clarification"
    intent = actual.get("intent")
    specialist = actual.get("specialist") or {}
    tool_calls = specialist.get("tool_calls", [])
    evidence = specialist.get("evidence", [])
    if intent == "unsupported":
        return "decline_out_of_scope"
    if not tool_calls:
        return "no_tool_call"
    if evidence:
        return "answer_with_evidence"
    response_text = " ".join(
        str(part) for part in [actual.get("answer"), specialist.get("answer")]
    ).casefold()
    if "no fee record" in response_text:
        return "report_no_record"
    if specialist.get("needs_follow_up"):
        return "ask_follow_up"
    return "tool_call_without_evidence"


def _action_check(case: EvaluationCase, actual: dict[str, Any]) -> bool:
    action = case.expected_action
    observed = _observed_action(actual)
    calls = actual["specialist"].get("tool_calls", [])
    evidence = actual["specialist"].get("evidence", [])
    if action in {"ask_for_missing_fields", "ask_for_valid_academic_year"}:
        clarification = actual.get("clarification") or {}
        question = str(clarification.get("question", "")).casefold()
        expected_fields = (case.model_extra.get("expected_clarification") or {}).get("fields", [])
        field_terms = {
            "board": ("board",),
            "standard": ("standard", "class"),
            "academic_year": ("academic year", "year"),
        }
        asks_for_expected_fields = all(
            any(term in question for term in field_terms.get(field, (field.casefold(),)))
            for field in expected_fields
        )
        return (
            bool(clarification)
            and actual.get("intent") == "clarification"
            and asks_for_expected_fields
        )
    if action == "confirm_board_candidate_before_lookup":
        clarification = actual.get("clarification") or {}
        expected_options = (case.model_extra.get("expected_clarification") or {}).get("options", [])
        candidates = [
            option
            for option in expected_options
            if option.casefold() not in {"cancel", "cancel lookup", "no"}
        ]
        return bool(clarification.get("options")) and all(
            option.casefold() in [str(value).casefold() for value in clarification["options"]]
            for option in candidates
        )
    if action == "report_no_record":
        return observed == action and bool(calls) and not evidence
    if action == "decline_out_of_scope":
        return actual.get("intent") == "unsupported" and not calls
    if action == "elicit_language_then_search":
        prompt = actual.get("mcp_elicitation") or {}
        return bool(prompt) and {"English", "Hindi"}.issubset(set(prompt.get("options", [])))
    if action == "cancel_without_search_or_title":
        return observed == action and not evidence
    if action == "skip_language_elicitation_then_search":
        return not actual.get("mcp_elicitation") and bool(evidence)
    if action == "answer_from_tool":
        return actual.get("intent") == "finance" and bool(calls) and bool(evidence)
    if action.startswith(("answer_", "search_and_answer_", "abstain_from_")):
        return bool(calls) and bool(evidence)
    return observed == action


def _case_result(case: EvaluationCase, actual: dict[str, Any]) -> dict[str, Any]:
    calls = actual["specialist"].get("tool_calls", [])
    actual_tool_names = [call.get("name") for call in calls]
    intent_correct = actual.get("intent") == case.expected_intent
    if case.expected_tool is None:
        tool_correct: bool | None = not calls
    else:
        tool_correct = case.expected_tool in actual_tool_names
    arguments_correct = _tool_argument_matches(case.expected_tool_arguments, calls)
    expected_language = case.expected_tool_arguments.get("language")
    if expected_language == "ask_user_to_choose":
        arguments_correct = bool(arguments_correct and actual.get("mcp_elicitation"))
    elif expected_language == "cancelled":
        arguments_correct = bool(
            arguments_correct
            and actual.get("mcp_elicitation", {}).get("resolution") == "cancelled_by_dataset"
        )
    action_correct = _action_check(case, actual)
    fee_values_match: bool | None = None
    if case.expected_fee is not None:
        evidence = " ".join(
            item.get("detail", "") for item in actual["specialist"].get("evidence", [])
        )
        expected_amounts = {
            amount
            for fee in case.expected_fee.values()
            for amount in re.findall(r"\d[\d,]*\.\d{2}", fee)
        }
        answer = str(actual.get("answer") or "")
        normalize = lambda text: {
            amount.replace(",", "")
            for amount in re.findall(r"\d[\d,]*\.\d{2}", text)
        }
        expected_normalized = {amount.replace(",", "") for amount in expected_amounts}
        evidence_amounts = normalize(evidence)
        answer_amounts = normalize(answer)
        requires_inr = any("INR" in fee.upper() for fee in case.expected_fee.values())
        fee_values_match = (
            bool(expected_normalized)
            and expected_normalized.issubset(evidence_amounts)
            and expected_normalized.issubset(answer_amounts)
            and answer_amounts.issubset(expected_normalized)
            and (not requires_inr or "INR" in answer.upper())
        )

    expected_prompt_actions = {
        "ask_for_missing_fields",
        "ask_for_valid_academic_year",
        "confirm_board_candidate_before_lookup",
        "elicit_language_then_search",
        "cancel_without_search_or_title",
    }
    clarification_expected = case.expected_action in expected_prompt_actions
    clarification_produced = bool(actual.get("clarification") or actual.get("mcp_elicitation"))
    forbidden_phrases = case.model_extra.get("expected_forbidden_phrases") or []
    answer_text = "\n".join(
        str(part)
        for part in [actual.get("answer"), (actual.get("specialist") or {}).get("answer")]
        if part
    ).casefold()
    prompt_disclosure_safe = (
        not any(str(phrase).casefold() in answer_text for phrase in forbidden_phrases)
        if forbidden_phrases
        else None
    )

    return {
        "id": case.id,
        "category": case.category,
        "user_input": case.user_input,
        "expected": {
            "intent": case.expected_intent,
            "action": case.expected_action,
            "tool": case.expected_tool,
            "tool_arguments": case.expected_tool_arguments,
            "fee": case.expected_fee,
            "reference": case.reference,
            "reference_contexts": case.reference_contexts,
        },
        "actual": actual,
        "checks": {
            "intent_correct": intent_correct,
            "tool_selection_correct": tool_correct,
            "tool_arguments_correct": arguments_correct,
            "expected_action_satisfied": action_correct,
            "fee_values_correct": fee_values_match,
            "clarification_expected": clarification_expected,
            "clarification_produced": clarification_produced,
            "prompt_disclosure_safe": prompt_disclosure_safe,
            "observed_action": _observed_action(actual),
        },
    }


def _accuracy(results: list[dict[str, Any]], check_name: str) -> dict[str, int | float | None]:
    checked = [result["checks"][check_name] for result in results]
    evaluated = [value for value in checked if value is not None]
    correct = sum(value is True for value in evaluated)
    return {
        "correct": correct,
        "total": len(evaluated),
        "accuracy": correct / len(evaluated) if evaluated else None,
    }


def _intent_classification_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute one-vs-rest metrics from categorical labels; no AUC without confidence."""
    labels = ["finance", "academic", "clarification", "unsupported"]
    predicted_labels = {result["actual"].get("intent") or "unknown" for result in results}
    columns = labels + sorted(predicted_labels - set(labels))
    matrix = {expected: {predicted: 0 for predicted in columns} for expected in labels}
    for result in results:
        expected = result["expected"]["intent"]
        predicted = result["actual"].get("intent") or "unknown"
        matrix.setdefault(expected, {label: 0 for label in columns})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1

    total = len(results)
    per_intent: dict[str, Any] = {}
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(matrix.get(other, {}).get(label, 0) for other in labels if other != label)
        fn = sum(matrix.get(label, {}).values()) - tp
        tn = total - tp - fp - fn
        per_intent[label] = {
            "support": tp + fn,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "tpr": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
        }
    macro_values = {
        key: [item[key] for item in per_intent.values() if item[key] is not None]
        for key in ("precision", "recall", "fpr")
    }
    return {
        "labels": labels,
        "confusion_matrix": matrix,
        "per_intent": per_intent,
        "macro_average": {
            key: sum(values) / len(values) if values else None
            for key, values in macro_values.items()
        },
        "note": "TPR equals recall. FPR is one-vs-rest. ROC AUC is unavailable because the supervisor emits no per-intent confidence scores.",
    }


def _workflow_quality_metrics(
    cases: list[EvaluationCase], results: list[dict[str, Any]]
) -> dict[str, Any]:
    results_by_id = {result["id"]: result for result in results}
    tool_cases = [case for case in cases if case.expected_tool]
    tool_successes = 0
    for case in tool_cases:
        result = results_by_id.get(case.id)
        calls = (result or {}).get("actual", {}).get("specialist", {}).get("tool_calls", [])
        if any(call.get("name") == case.expected_tool for call in calls):
            tool_successes += 1

    clarification_cases = [
        result for result in results if result["checks"].get("clarification_expected")
    ]
    no_clarification_cases = [
        result for result in results if result["checks"].get("clarification_expected") is False
    ]
    unnecessary = sum(
        result["checks"].get("clarification_produced", False)
        for result in no_clarification_cases
    )
    correct_clarifications = sum(
        result["checks"].get("expected_action_satisfied") is True
        for result in clarification_cases
    )
    return {
        "tool_success_rate": {
            "successful_calls": tool_successes,
            "expected_tool_cases": len(tool_cases),
            "rate": tool_successes / len(tool_cases) if tool_cases else None,
            "definition": "Expected tool cases whose validated call returned and was recorded by the specialist.",
        },
        "clarification_accuracy": {
            "correct": correct_clarifications,
            "expected_clarification_cases": len(clarification_cases),
            "rate": correct_clarifications / len(clarification_cases) if clarification_cases else None,
        },
        "unnecessary_clarification_rate": {
            "unnecessary": unnecessary,
            "cases_not_expecting_clarification": len(no_clarification_cases),
            "rate": unnecessary / len(no_clarification_cases) if no_clarification_cases else None,
        },
    }


def _latency_percentile(results: list[dict[str, Any]], percentile: int) -> float | None:
    durations = sorted(float(result["actual"]["duration_seconds"]) for result in results)
    if not durations:
        return None
    index = max(0, (percentile * len(durations) + 99) // 100 - 1)
    return durations[index]


async def run_evaluation(
    cases: list[EvaluationCase], *, timeout_seconds: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    graph = _make_graph()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for case in cases:
        try:
            actual = await _invoke_case(graph, case, timeout_seconds)
            results.append(_case_result(case, actual))
        except Exception as error:
            errors.append({"id": case.id, "error": f"{type(error).__name__}: {error}"})
    return results, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-id", action="append", help="Run a selected case; repeatable")
    parser.add_argument("--limit", type=int, help="Run only the first N cases")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Score academic RAG cases with Ragas using local Ollama models",
    )
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.id in selected]
        missing = selected - {case.id for case in cases}
        if missing:
            parser.error(f"Unknown case ID(s): {', '.join(sorted(missing))}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        cases = cases[: args.limit]
    if not cases:
        parser.error("No evaluation cases selected")

    results, errors = asyncio.run(
        run_evaluation(cases, timeout_seconds=args.timeout_seconds)
    )
    settings = get_settings()
    ragas_summary = None
    if args.ragas:
        from support_agents.evaluation.ragas import RagasDependencyError, score_rag_results

        try:
            ragas_summary = asyncio.run(score_rag_results(results, settings))
        except RagasDependencyError as error:
            parser.error(str(error))

    summary = {
        "cases_requested": len(cases),
        "cases_completed": len(results),
        "cases_errored": len(errors),
        "intent_accuracy": _accuracy(results, "intent_correct"),
        "tool_selection_accuracy": _accuracy(results, "tool_selection_correct"),
        "tool_argument_accuracy": _accuracy(results, "tool_arguments_correct"),
        "expected_action_accuracy": _accuracy(results, "expected_action_satisfied"),
        "fee_value_accuracy": _accuracy(results, "fee_values_correct"),
        "intent_classification": _intent_classification_metrics(results),
        "workflow_quality": _workflow_quality_metrics(cases, results),
        "prompt_injection_safety": _accuracy(results, "prompt_disclosure_safe"),
        "latency_seconds": {
            "p50": _latency_percentile(results, 50),
            "p95": _latency_percentile(results, 95),
        },
        "ragas": ragas_summary
        or {"enabled": False, "note": "Pass --ragas to score academic RAG cases."},
        "note": "RAG context and answer quality are scored only when --ragas is enabled.",
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "configured_providers": {
            "supervisor": settings.supervisor_provider,
            "final_responder": settings.final_response_provider,
            "academic_agent": "ollama",
        },
        "summary": summary,
        "errors": errors,
        "results": results,
    }
    output = args.output or (
        PROJECT_ROOT
        / "outputs"
        / "evaluations"
        / f"support-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
