"""HTTP-level tests for the FastAPI app using a fake graph (no models or database)."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from support_agents.api import create_app


class FakeGraph:
    def __init__(self, result: dict | Exception) -> None:
        self.result = result
        self.calls: list[tuple[object, dict]] = []

    async def ainvoke(self, payload, config):
        self.calls.append((payload, config))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _client(graph: FakeGraph) -> TestClient:
    return TestClient(create_app(graph=graph))


def test_health_and_home_page() -> None:
    with _client(FakeGraph({})) as client:
        assert client.get("/health").json() == {"status": "ok"}
        home = client.get("/")
        assert home.status_code == 200
        assert "/static/app.js" in home.text
        assert client.get("/static/app.js").status_code == 200


def test_query_completed() -> None:
    graph = FakeGraph(
        {"answer": "61,500.00 INR", "intent": "finance", "limitations": ["demo data"]}
    )
    with _client(graph) as client:
        response = client.post("/api/v1/query", json={"query": " fees? "})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "completed"
    assert body["answer"] == "61,500.00 INR"
    payload, config = graph.calls[0]
    assert payload["query"] == "fees?"
    assert payload["user_id"] == "anonymous"
    assert payload["allow_mcp_elicitation"] is False
    assert config["configurable"]["thread_id"] == body["request_id"]


def test_query_needs_input() -> None:
    pending = SimpleNamespace(value={"question": "Which board?", "options": ["CBSE"]})
    with _client(FakeGraph({"__interrupt__": [pending]})) as client:
        body = client.post("/api/v1/query", json={"query": "fees?"}).json()
    assert body["status"] == "needs_input"
    assert body["question"] == "Which board?"
    assert body["options"] == ["CBSE"]


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"content": "{}", "headers": {"content-type": "text/plain"}}, 415),
        ({"content": "not json", "headers": {"content-type": "application/json"}}, 400),
        ({"json": {"query": ""}}, 400),
        ({"json": {"query": "x", "extra": 1}}, 400),
        ({"json": {"query": "x", "request_id": "nope"}}, 400),
    ],
)
def test_query_rejects_bad_requests(kwargs: dict, status: int) -> None:
    with _client(FakeGraph({})) as client:
        response = client.post("/api/v1/query", **kwargs)
    assert response.status_code == status
    assert "error" in response.json()
    assert "request_id" in response.json()


def test_query_graph_failure_returns_502() -> None:
    with _client(FakeGraph(RuntimeError("boom"))) as client:
        response = client.post("/api/v1/query", json={"query": "fees?"})
    assert response.status_code == 502
    assert "boom" not in response.text


def test_resume_validates_id_and_maps_failure_to_409() -> None:
    with _client(FakeGraph(RuntimeError("expired"))) as client:
        bad_id = client.post("/api/v1/query/not-a-uuid/resume", json={"answer": "x"})
        assert bad_id.status_code == 400
        response = client.post(f"/api/v1/query/{uuid4()}/resume", json={"answer": "CBSE"})
    assert response.status_code == 409


def test_elicitation_endpoints() -> None:
    request_id = uuid4()
    with _client(FakeGraph({})) as client:
        assert client.get(f"/api/v1/elicitation/{request_id}").json()["status"] == "waiting"
        assert client.get("/api/v1/elicitation/bad").status_code == 400
        missing = client.post(
            f"/api/v1/elicitation/{request_id}/answer", json={"answer": "English"}
        )
    assert missing.status_code == 404


def test_ingest_requires_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(academic_ingest_api_key=None)
    monkeypatch.setattr("support_agents.api.get_settings", lambda: settings)
    with _client(FakeGraph({})) as client:
        assert client.post("/api/v1/academic/ingest").status_code == 503
        settings.academic_ingest_api_key = "secret"
        assert client.post("/api/v1/academic/ingest").status_code == 401
        wrong = client.post("/api/v1/academic/ingest", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        bad_type = client.post(
            "/api/v1/academic/ingest",
            headers={"Authorization": "Bearer secret"},
            json={"document_type": "other"},
        )
        assert bad_type.status_code == 400
