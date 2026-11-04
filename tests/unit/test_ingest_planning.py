"""Pure tests for deciding which chunks need a new embedding."""

from __future__ import annotations

from support_agents.db.ingest_academic_pdf import (
    DocumentChunk,
    ExistingChunk,
    chunk_hash,
    plan_changes,
)

MODEL = "model-a"


def _chunks(*texts: str) -> list[DocumentChunk]:
    return [DocumentChunk(text, page=1, standard=None) for text in texts]


def _stored(*texts: str, model: str = MODEL) -> list[ExistingChunk]:
    return [ExistingChunk(100 + i, chunk_hash(text), model) for i, text in enumerate(texts)]


def test_unchanged_document_embeds_nothing() -> None:
    plan = plan_changes(_stored("a", "b", "c"), _chunks("a", "b", "c"), MODEL)
    assert plan.to_embed == [] and plan.inserts == [] and plan.to_delete == []
    assert plan.kept == [(100, 0), (101, 1), (102, 2)]


def test_edited_chunk_is_the_only_one_embedded() -> None:
    plan = plan_changes(_stored("a", "b", "c"), _chunks("a", "B!", "c"), MODEL)
    assert plan.to_embed == ["B!"]
    assert plan.inserts == [1]
    assert plan.to_delete == [101]
    assert plan.kept == [(100, 0), (102, 2)]


def test_insert_in_the_middle_only_renumbers_later_chunks() -> None:
    plan = plan_changes(_stored("a", "b", "c"), _chunks("a", "new", "b", "c"), MODEL)
    assert plan.to_embed == ["new"]
    assert plan.to_delete == []
    assert plan.kept == [(100, 0), (101, 2), (102, 3)]


def test_removed_chunk_is_deleted_without_embedding() -> None:
    plan = plan_changes(_stored("a", "b", "c"), _chunks("a", "c"), MODEL)
    assert plan.to_embed == []
    assert plan.to_delete == [101]
    assert plan.kept == [(100, 0), (102, 1)]


def test_repeated_text_is_matched_one_row_per_chunk() -> None:
    plan = plan_changes(_stored("x", "x"), _chunks("x", "x", "x"), MODEL)
    assert plan.kept == [(100, 0), (101, 1)]
    assert plan.inserts == [2]
    assert plan.to_embed == ["x"]  # embedded once even if several chunks share the text


def test_new_duplicate_texts_are_embedded_once() -> None:
    plan = plan_changes([], _chunks("y", "y"), MODEL)
    assert plan.to_embed == ["y"]
    assert plan.inserts == [0, 1]


def test_different_embedding_model_reembeds_everything() -> None:
    plan = plan_changes(_stored("a", "b", model="old-model"), _chunks("a", "b"), MODEL)
    assert plan.to_embed == ["a", "b"]
    assert plan.kept == []
    assert plan.to_delete == [100, 101]


def test_rows_without_a_hash_are_replaced() -> None:
    legacy = [ExistingChunk(1, None, MODEL)]
    plan = plan_changes(legacy, _chunks("a"), MODEL)
    assert plan.to_embed == ["a"] and plan.to_delete == [1]


def test_force_reembeds_everything() -> None:
    plan = plan_changes(_stored("a", "b"), _chunks("a", "b"), MODEL, force=True)
    assert plan.to_embed == ["a", "b"] and plan.kept == []


def test_hash_matches_sql_backfill_definition() -> None:
    # Migration 005 uses encode(sha256(convert_to(content, 'UTF8')), 'hex').
    assert chunk_hash("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
