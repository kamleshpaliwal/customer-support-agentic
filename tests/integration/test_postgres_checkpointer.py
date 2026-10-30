"""A paused clarification must survive a restart (new saver/pool on the same database)."""

from __future__ import annotations

import asyncio
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from langgraph.types import Command

from support_agents.checkpointer import close_checkpointer, create_postgres_checkpointer
from support_agents.graph import GraphDependencies, build_support_graph
from support_agents.schemas import Intent, IntentDecision, SpecialistResult
from support_agents.settings import get_settings


def _database_reachable() -> bool:
    try:
        url = urlparse(get_settings().database_url)
        with socket.create_connection((url.hostname, url.port or 5432), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_reachable(), reason="PostgreSQL is not reachable")


class _Detector:
    """Asks for clarification first, then routes to finance once the user has answered."""

    async def detect(self, query: str) -> IntentDecision:
        if "User clarification" in query:
            return IntentDecision(intent=Intent.UNSUPPORTED, rationale="test")
        return IntentDecision(
            intent=Intent.CLARIFICATION,
            rationale="test",
            clarification_question="Which board?",
        )


class _Specialist:
    async def respond(self, query, *, user_id, fee_lookup=None, request_id=None):
        return SpecialistResult(specialist="finance", answer="unused")


class _Responder:
    async def render(self, query, result):
        return f"done: {result.answer}"


def _graph(checkpointer):
    return build_support_graph(
        GraphDependencies(_Detector(), _Specialist(), _Specialist(), _Responder()),
        checkpointer=checkpointer,
    )


def test_paused_run_resumes_with_a_fresh_checkpointer() -> None:
    asyncio.run(_pause_then_resume_after_restart())


async def _pause_then_resume_after_restart() -> None:
    url = get_settings().database_url
    thread = {"configurable": {"thread_id": str(uuid4())}}
    state = {"request_id": "r", "user_id": "u", "query": "fees?"}

    saver_a = await create_postgres_checkpointer(url)
    paused = await _graph(saver_a).ainvoke(state, config=thread)
    assert paused["__interrupt__"][0].value["question"] == "Which board?"
    await close_checkpointer(saver_a)

    # Simulates a process restart: brand-new pool and saver, same thread id.
    saver_b = await create_postgres_checkpointer(url)
    resumed = await _graph(saver_b).ainvoke(Command(resume="CBSE"), config=thread)
    await close_checkpointer(saver_b)
    assert resumed["answer"].startswith("done:")
