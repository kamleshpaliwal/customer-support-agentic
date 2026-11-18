"""Pure parts of retrieval: query building, rank fusion and reranker ordering."""

import asyncio

import pytest

from support_agents.retrieval import (
    keyword_tsquery,
    order_by_scores,
    reciprocal_rank_fusion,
    search_passages,
)
from support_agents.schemas import AcademicSearchMatch


def match(content: str, page: int = 1, similarity: float = 0.8) -> AcademicSearchMatch:
    return AcademicSearchMatch(
        content=content, source="guide", page=page, similarity_score=similarity
    )


def test_keyword_tsquery_ors_words_and_keeps_single_digits():
    assert keyword_tsquery("What subjects in Standard 6?") == "what | subjects | in | standard | 6"


def test_keyword_tsquery_drops_single_letters_and_duplicates():
    assert keyword_tsquery("a math math b 10") == "math | 10"


def test_keyword_tsquery_is_none_without_words():
    assert keyword_tsquery("?! -") is None


def test_rrf_prefers_chunks_found_by_both_lists():
    a, b, c = match("a"), match("b"), match("c")
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert [m.content for m in fused] == ["b", "a", "c"]


def test_rrf_with_one_list_keeps_its_order():
    items = [match("a"), match("b"), match("c")]
    assert reciprocal_rank_fusion([items]) == items


def test_rrf_treats_same_text_on_different_pages_as_different_chunks():
    fused = reciprocal_rank_fusion([[match("same", page=1)], [match("same", page=2)]])
    assert len(fused) == 2


def test_order_by_scores_sorts_descending_and_keeps_ties_stable():
    items = [match("a"), match("b"), match("c"), match("d")]
    ordered = order_by_scores(items, [0.1, 0.9, 0.1, 0.5])
    assert [m.content for m in ordered] == ["b", "d", "a", "c"]


class _Embedder:
    model = "fake"

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 768


@pytest.fixture
def stage_one(monkeypatch):
    candidates = [
        match("a", similarity=0.9),
        match("b", similarity=0.8),
        match("c", similarity=0.1),
    ]

    async def fake_vector(pool, *, vector, filters, limit):
        return candidates[:limit]

    monkeypatch.setattr("support_agents.retrieval.vector_candidates", fake_vector)


class _Reranker:
    def __init__(self, scores=None, error=None):
        self.scores, self.error, self.seen = scores, error, None

    async def score(self, query, passages):
        self.seen = passages
        if self.error:
            raise self.error
        return self.scores


def _search(**kwargs):
    return asyncio.run(
        search_passages(
            None, _Embedder(), query="q", document_type="syllabus", limit=2, mode="vector",
            **kwargs,
        )
    )


def test_rerank_reorders_after_the_similarity_threshold(stage_one):
    reranker = _Reranker(scores=[0.2, 0.7])
    result = _search(rerank=True, reranker=reranker)
    assert reranker.seen == ["a", "b"]  # "c" is below the similarity threshold
    assert [m.content for m in result] == ["b", "a"]


def test_rerank_failure_keeps_first_stage_order(stage_one):
    result = _search(rerank=True, reranker=_Reranker(error=RuntimeError("boom")))
    assert [m.content for m in result] == ["a", "b"]


def test_no_rerank_means_reranker_is_not_called(stage_one):
    reranker = _Reranker(scores=[0.0, 1.0])
    result = _search(rerank=False, reranker=reranker)
    assert reranker.seen is None
    assert [m.content for m in result] == ["a", "b"]
