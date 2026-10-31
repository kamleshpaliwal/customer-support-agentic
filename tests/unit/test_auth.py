"""Authentication and authorization: token checks, groups, and per-user session ownership."""

from __future__ import annotations

import time
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from support_agents.api import create_app
from support_agents.auth import TokenVerifier

ISSUER = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TEST"
CLIENT_ID = "test-client"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


async def _key_provider(token: str):
    return _KEY.public_key()


def _token(
    *,
    sub: str = "user-1",
    groups: tuple[str, ...] = ("staff",),
    key=_KEY,
    **overrides,
) -> str:
    claims = {
        "sub": sub,
        "iss": ISSUER,
        "client_id": CLIENT_ID,
        "token_use": "access",
        "cognito:groups": list(groups),
        "exp": int(time.time()) + 600,
    }
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm="RS256")


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, payload, config):
        self.calls.append(payload)
        return {"answer": "ok", "intent": "finance", "limitations": []}


@pytest.fixture
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def client(graph: FakeGraph):
    verifier = TokenVerifier(issuer=ISSUER, client_id=CLIENT_ID, key_provider=_key_provider)
    with TestClient(create_app(graph=graph, auth=verifier)) as test_client:
        yield test_client


def _query(client: TestClient, token: str | None, **body):
    headers = _headers(token) if token else {}
    return client.post("/api/v1/query", json={"query": "fees?", **body}, headers=headers)


def test_missing_token_is_rejected(client: TestClient) -> None:
    response = _query(client, None)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "error" in response.json()


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("not-a-jwt", id="garbage"),
        pytest.param(_token(exp=int(time.time()) - 10), id="expired"),
        pytest.param(_token(iss="https://evil.example.com"), id="wrong-issuer"),
        pytest.param(_token(client_id="other-client"), id="wrong-client"),
        pytest.param(_token(token_use="id", aud=CLIENT_ID, client_id=None), id="id-token"),
        pytest.param(_token(key=_OTHER_KEY), id="wrong-signature"),
    ],
)
def test_invalid_tokens_are_rejected(client: TestClient, token: str) -> None:
    assert _query(client, token).status_code == 401


def test_user_outside_allowed_groups_is_forbidden(client: TestClient) -> None:
    assert _query(client, _token(groups=("parent",))).status_code == 403
    assert _query(client, _token(groups=())).status_code == 403


def test_identity_comes_from_the_token_not_the_body(client: TestClient, graph: FakeGraph) -> None:
    response = _query(client, _token(sub="staff-42"), user_id="someone-else")
    assert response.status_code == 200
    assert graph.calls[0]["user_id"] == "staff-42"


def test_users_cannot_touch_each_others_sessions(client: TestClient) -> None:
    request_id = str(uuid4())
    alice, bob = _token(sub="alice"), _token(sub="bob")
    assert _query(client, alice, request_id=request_id).status_code == 200

    # Bob can neither reuse, resume, poll nor answer Alice's request.
    assert _query(client, bob, request_id=request_id).status_code == 404
    resume = client.post(
        f"/api/v1/query/{request_id}/resume", json={"answer": "x"}, headers=_headers(bob)
    )
    assert resume.status_code == 404
    poll = client.get(f"/api/v1/elicitation/{request_id}", headers=_headers(bob))
    assert poll.status_code == 404
    answer = client.post(
        f"/api/v1/elicitation/{request_id}/answer",
        json={"answer": "English"},
        headers=_headers(bob),
    )
    assert answer.status_code == 404

    # Alice still can.
    own = client.get(f"/api/v1/elicitation/{request_id}", headers=_headers(alice))
    assert own.status_code == 200


def test_ingest_requires_admin_group(client: TestClient) -> None:
    assert client.post("/api/v1/academic/ingest").status_code == 401
    staff = client.post("/api/v1/academic/ingest", headers=_headers(_token(groups=("staff",))))
    assert staff.status_code == 403


def test_public_routes_stay_open(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/v1/auth/config").json()["enabled"] is True


def test_allowed_groups_are_configurable() -> None:
    verifier = TokenVerifier(
        issuer=ISSUER,
        client_id=CLIENT_ID,
        key_provider=_key_provider,
        allowed_groups=frozenset({"staff", "admin", "parent"}),
    )
    with TestClient(create_app(graph=FakeGraph(), auth=verifier)) as client:
        assert _query(client, _token(groups=("parent",))).status_code == 200


def test_auth_disabled_keeps_local_development_open() -> None:
    with TestClient(create_app(graph=FakeGraph())) as client:
        assert client.get("/api/v1/auth/config").json()["enabled"] is False
        assert _query(client, None).status_code == 200


def test_app_refuses_to_start_without_auth_settings() -> None:
    from types import SimpleNamespace

    settings = SimpleNamespace(auth_issuer=None, auth_client_id=None)
    with pytest.raises(ValueError, match="AUTH_ISSUER and AUTH_CLIENT_ID"):
        TokenVerifier.from_settings(settings)  # type: ignore[arg-type]
