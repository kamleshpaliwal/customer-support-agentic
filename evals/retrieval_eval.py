"""Retrieval-only evaluation: did the search return the expected pages, and how high?

Unlike ``run_eval.py`` this calls ``search_passages`` directly, with no LLM involved, so it is fast
and deterministic and isolates the effect of retrieval changes. Metrics per case:

* hit@k: any of the first k passages overlaps an expected page.
* reciprocal rank: 1 / rank of the first overlapping passage (0 if none); MRR is the mean.

Run with PostgreSQL and Ollama up:
  python evals/retrieval_eval.py --strategy all            # compare the four strategies
  python evals/retrieval_eval.py --query "..." [--type book_recommendations --language English]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from support_agents.db.connection import create_pool
from support_agents.embeddings import Embedder, create_embedder
from support_agents.reranking import get_reranker
from support_agents.retrieval import search_passages
from support_agents.schemas import AcademicSearchMatch

DATASET = Path(__file__).parent / "datasets" / "support_eval.jsonl"
DOCUMENT_TYPES = {
    "search_syllabus": "syllabus",
    "search_book_recommendations": "book_recommendations",
}
LANGUAGES = {"English", "Hindi"}


def load_cases(dataset: Path = DATASET) -> list[dict[str, Any]]:
    """Academic cases that name expected pages and (for books) a concrete language."""
    cases = []
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        document_type = DOCUMENT_TYPES.get(case.get("expected_tool") or "")
        if document_type is None or not case.get("expected_pages"):
            continue
        language = (case.get("expected_tool_arguments") or {}).get("language")
        if document_type == "book_recommendations" and language not in LANGUAGES:
            continue  # the language is chosen by the user at run time; nothing to replay
        cases.append(
            {
                "id": case["id"],
                "query": case["user_input"],
                "document_type": document_type,
                "language": language if document_type == "book_recommendations" else None,
                "expected_pages": set(case["expected_pages"]),
            }
        )
    return cases


def overlaps(match: AcademicSearchMatch, expected_pages: set[int]) -> bool:
    if match.page is None:
        return False
    return any(match.page <= page <= (match.page_end or match.page) for page in expected_pages)


def first_relevant_rank(matches: list[AcademicSearchMatch], expected_pages: set[int]) -> int | None:
    for rank, match in enumerate(matches, start=1):
        if overlaps(match, expected_pages):
            return rank
    return None


STRATEGIES = {
    "vector": ("vector", False),
    "hybrid": ("hybrid", False),
    "vector+rerank": ("vector", True),
    "hybrid+rerank": ("hybrid", True),
}


async def run_strategy(
    pool: Any, embedder: Embedder, cases: list[dict[str, Any]], k: int, strategy: str
) -> dict[str, Any]:
    mode, rerank = STRATEGIES[strategy]
    results = []
    for case in cases:
        started = time.perf_counter()
        matches = await search_passages(
            pool,
            embedder,
            query=case["query"],
            document_type=case["document_type"],
            language=case["language"],
            limit=k,
            mode=mode,
            rerank=rerank,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        rank = first_relevant_rank(matches, case["expected_pages"])
        results.append(
            {
                "id": case["id"],
                "rank": rank,
                "returned": len(matches),
                "hit": rank is not None,
                "reciprocal_rank": 1 / rank if rank else 0.0,
                "latency_ms": round(latency_ms, 1),
                "pages": [[m.page, m.page_end] for m in matches],
                "expected_pages": sorted(case["expected_pages"]),
            }
        )
    count = len(results)
    return {
        "strategy": strategy,
        "k": k,
        "cases": count,
        f"hit@{k}": round(sum(r["hit"] for r in results) / count, 3),
        "hit@1": round(sum(r["rank"] == 1 for r in results) / count, 3),
        "mrr": round(sum(r["reciprocal_rank"] for r in results) / count, 3),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / count, 1),
        "results": results,
    }


async def evaluate(k: int, strategies: list[str], dataset: Path) -> list[dict[str, Any]]:
    cases = load_cases(dataset)
    embedder = create_embedder()
    pool = await create_pool()
    try:
        reports = []
        for strategy in strategies:
            if STRATEGIES[strategy][1]:  # load the reranker once, outside the timed queries
                await get_reranker().warm_up()
            reports.append(await run_strategy(pool, embedder, cases, k, strategy))
        return reports
    finally:
        await pool.close()


async def show_query(query: str, document_type: str, language: str | None, k: int) -> None:
    """Print the passages each strategy returns for one query, to see how the stages differ."""
    embedder = create_embedder()
    pool = await create_pool()
    try:
        for strategy, (mode, rerank) in STRATEGIES.items():
            if rerank:
                await get_reranker().warm_up()
            matches = await search_passages(
                pool,
                embedder,
                query=query,
                document_type=document_type,  # type: ignore[arg-type]
                language=language,
                limit=k,
                mode=mode,
                rerank=rerank,
            )
            print(f"\n== {strategy}")
            for rank, match in enumerate(matches, start=1):
                snippet = " ".join(match.content.split())[:90]
                print(
                    f"{rank}. p{match.page} {match.standard or '-'} "
                    f"cos={match.similarity_score:.3f}  {snippet}"
                )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=4, help="passages returned per query")
    parser.add_argument(
        "--strategy", choices=[*STRATEGIES, "all"], default="vector", help="retrieval strategy"
    )
    parser.add_argument("--dataset", type=Path, default=DATASET, help="JSONL eval cases")
    parser.add_argument("--json", type=Path, help="write the full report to this file")
    parser.add_argument("--query", help="show what each strategy returns for one query")
    parser.add_argument("--type", choices=sorted(DOCUMENT_TYPES.values()), default="syllabus")
    parser.add_argument("--language", choices=sorted(LANGUAGES))
    args = parser.parse_args()

    if args.query:
        asyncio.run(show_query(args.query, args.type, args.language, args.k))
        return

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    reports = asyncio.run(evaluate(args.k, strategies, args.dataset))
    for report in reports:
        print(f"\n== {report['strategy']}")
        for result in report["results"]:
            if result["hit"] and result["rank"] == 1:
                continue  # only list cases that were not perfect
            mark = "ok  " if result["hit"] else "MISS"
            print(
                f"{mark} {result['id']:<9} rank={result['rank']} "
                f"got={result['pages']} want={result['expected_pages']}"
            )
    metrics = f"hit@{args.k}"
    print(f"\n{'strategy':<15} {metrics:>6} {'hit@1':>6} {'mrr':>6} {'ms/query':>9}")
    for report in reports:
        print(
            f"{report['strategy']:<15} {report[metrics]:>6} {report['hit@1']:>6} "
            f"{report['mrr']:>6} {report['mean_latency_ms']:>9}"
        )
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
