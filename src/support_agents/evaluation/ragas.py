"""Optional Ragas scoring for academic retrieval and answer quality."""

from __future__ import annotations

import logging
from typing import Any

from support_agents.settings import Settings


RAG_TOOL_NAMES = {"search_syllabus", "search_book_recommendations"}
METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)
MAX_CONTEXT_CHARS = 5000
LOGGER = logging.getLogger(__name__)


class RagasDependencyError(RuntimeError):
    """Raised when Ragas evaluation is requested without its optional dependencies."""


def _metric_inputs(
    name: str,
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
) -> dict[str, Any]:
    if name in {"context_precision", "context_recall"}:
        return {
            "user_input": user_input,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
        }
    if name == "faithfulness":
        return {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }
    return {"user_input": user_input, "response": response}


def _mean_metric(scores: list[float]) -> float | None:
    return round(sum(scores) / len(scores), 4) if scores else None


async def score_rag_results(
    results: list[dict[str, Any]], settings: Settings
) -> dict[str, Any]:
    """Attach per-case Ragas scores and return metric-level aggregates.

    Ragas uses Ollama's OpenAI-compatible local endpoint, so this path does not
    require OpenAI or Bedrock credentials. Metric calls run sequentially to limit
    local load.
    """
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as error:
        raise RagasDependencyError(
            "Could not import the optional Ragas evaluation dependencies. Reinstall the "
            'compatible evaluation extras with: python -m pip install -e ".[eval]". '
            f"Import error: {error}"
        ) from error

    # Ragas 0.4 collection metrics require its modern Instructor LLM and
    # embedding interfaces. Ollama exposes compatible chat and embedding APIs
    # under /v1, so both can remain local without separate provider credentials.
    ollama_client = AsyncOpenAI(
        api_key="ollama",
        base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
        timeout=settings.ollama_request_timeout_seconds,
    )
    evaluator_llm = llm_factory(
        settings.academic_agent_ollama_model,
        provider="openai",
        client=ollama_client,
        temperature=0,
        max_tokens=1024,
        # Qwen's hidden reasoning tokens are wasteful for these short,
        # schema-constrained scoring judgments and can dominate local eval time.
        extra_body={"think": False},
    )
    evaluator_embeddings = embedding_factory(
        "openai",
        model=settings.ollama_embedding_model,
        client=ollama_client,
        interface="modern",
    )
    metrics: dict[str, Any] = {
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
    }
    score_values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    scored_cases = 0
    partial_cases = 0
    skipped_cases = 0

    for case_result in results:
        expected = case_result.get("expected", {})
        if expected.get("tool") not in RAG_TOOL_NAMES:
            continue

        reference_contexts = expected.get("reference_contexts") or []
        if not reference_contexts:
            case_result["ragas"] = {
                "status": "skipped",
                "reason": "No human-labeled reference contexts for this case.",
            }
            skipped_cases += 1
            continue

        actual = case_result.get("actual", {})
        if actual.get("mcp_elicitation") or actual.get("clarification"):
            case_result["ragas"] = {
                "status": "skipped",
                "reason": "No final answer was produced; the request is waiting for user input.",
            }
            skipped_cases += 1
            continue

        specialist = actual.get("specialist") or {}
        retrieved_contexts = [
            # Preserve citations in the judge's input. The final answer may cite
            # a PDF page, while the evidence detail alone contains only its text.
            f"[Source: {item.get('source', 'retrieved document')}] "
            f"{str(item['detail'])[:MAX_CONTEXT_CHARS]}"
            for item in specialist.get("evidence", [])
            if item.get("detail")
        ]
        response = (
            actual.get("answer")
            or (actual.get("clarification") or {}).get("question")
            or specialist.get("answer")
            or ""
        )
        user_input = str(case_result.get("user_input", ""))
        reference = str(expected.get("reference", ""))
        case_scores: dict[str, Any] = {}
        abstention_case = expected.get("action") == "abstain_from_inventing_title"
        scope_limit_case = (
            expected.get("action") == "answer_with_scope_limit_or_refer_to_school"
        )
        metrics_not_applicable: dict[str, str] = {}
        if abstention_case:
            # A missing title is a negative fact: the retrieved section can be
            # relevant without containing a literal sentence saying no title exists.
            # Standard recall/faithfulness judges can therefore penalize a correct abstention.
            metrics_not_applicable = {
                "context_recall": (
                    "This case tests whether a title is absent; the reference is not a positive "
                    "fact that must appear verbatim in a retrieved passage. Use the separate "
                    "abstention_correct check."
                ),
                "faithfulness": (
                    "The answer's key claim is that no title is named, which is inferred from "
                    "the relevant section's contents. The standard claim-to-context judge is not "
                    "reliable for this negative-fact case; use abstention_correct instead."
                ),
            }
        if scope_limit_case:
            # AnswerRelevancy intentionally scores zero when its question-generation
            # judge marks every generated question as noncommittal. A correct answer
            # that explains the guide lacks an exact syllabus can trigger that rule,
            # so evaluate it with the explicit scope-limit check instead.
            metrics_not_applicable["answer_relevancy"] = (
                "The expected answer is a grounded scope limitation, not the requested "
                "syllabus itself. Ragas AnswerRelevancy treats such correct noncommittal "
                "answers as zero; use scope_limit_answer_correct for this case."
            )
        if retrieved_contexts:
            applicable_names = [
                name for name in metrics
                if not (abstention_case and name in {"context_recall", "faithfulness"})
                and not (scope_limit_case and name == "answer_relevancy")
            ]
        else:
            applicable_names = ["answer_relevancy"]
            metrics_not_applicable.update({
                name: "No retrieved evidence was captured for this case."
                for name in metrics
                if name != "answer_relevancy"
            })
        applicable_metrics = [(name, metrics[name]) for name in applicable_names]
        applicable_metric_names = [name for name, _ in applicable_metrics]
        case_id = case_result.get("id", "unknown")

        for name in applicable_metric_names:
            metric = metrics[name]
            try:
                result = await metric.ascore(
                    **_metric_inputs(
                        name,
                        user_input=user_input,
                        response=str(response),
                        retrieved_contexts=retrieved_contexts,
                        reference=reference,
                    )
                )
                value = float(result.value)
                case_scores[name] = {
                    "score": round(value, 4),
                    "reason": getattr(result, "reason", None),
                }
                score_values[name].append(value)
                LOGGER.info("Ragas scored case=%s metric=%s score=%.4f", case_id, name, value)
            except Exception as error:
                case_scores[name] = {
                    "error": f"{type(error).__name__}: {error}"
                }
                LOGGER.warning(
                    "Ragas scoring failed case=%s metric=%s error=%s",
                    case_id,
                    name,
                    type(error).__name__,
                )

        successful_scores = sum("score" in item for item in case_scores.values())
        if successful_scores:
            scored_cases += 1
            is_partial = successful_scores != len(applicable_metric_names)
            partial_cases += is_partial
            case_result["ragas"] = {
                "status": "partial" if is_partial else "completed",
                "retrieved_context_count": len(retrieved_contexts),
                "metrics_applicable": applicable_metric_names,
                "metrics_not_applicable": metrics_not_applicable,
                "metrics": case_scores,
            }
        else:
            skipped_cases += 1
            case_result["ragas"] = {
                "status": "skipped",
                "reason": "No Ragas metric completed successfully.",
                "retrieved_context_count": len(retrieved_contexts),
                "metrics_applicable": applicable_metric_names,
                "metrics_not_applicable": metrics_not_applicable,
                "metrics": case_scores,
            }

    return {
        "enabled": True,
        "evaluator_provider": "ollama",
        "evaluator_model": settings.academic_agent_ollama_model,
        "embedding_model": settings.ollama_embedding_model,
        "cases_scored": scored_cases,
        "cases_partial": partial_cases,
        "cases_skipped": skipped_cases,
        "metrics": {
            name: {
                "mean": _mean_metric(score_values[name]),
                "scored_cases": len(score_values[name]),
            }
            for name in METRIC_NAMES
        },
        "note": (
            "Ragas scores are LLM-judge estimates; review per-case scores and reasons, "
            "especially for this small initial labeled dataset."
        ),
    }
