"""JWT bearer authentication and group-based authorization.

The identity provider (Amazon Cognito by default) owns sign-up and login. This module only
verifies the access token it issues: signature, expiry, issuer, app client and group.
Identity always comes from the verified token, never from the request body.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Request

from support_agents.settings import Settings

KeyProvider = Callable[[str], Awaitable[object]]


@dataclass(frozen=True)
class User:
    id: str
    groups: frozenset[str] = frozenset()
    is_admin: bool = False


ANONYMOUS = User(id="anonymous", is_admin=True)


def _split(value: str) -> frozenset[str]:
    return frozenset(item.strip() for item in value.split(",") if item.strip())


class TokenVerifier:
    def __init__(
        self,
        *,
        issuer: str,
        client_id: str,
        jwks_url: str | None = None,
        allowed_groups: frozenset[str] = frozenset({"staff", "admin"}),
        admin_group: str = "admin",
        key_provider: KeyProvider | None = None,
    ) -> None:
        self.issuer = issuer.rstrip("/")
        self.client_id = client_id
        self.allowed_groups = allowed_groups
        self.admin_group = admin_group
        if key_provider is None:
            client = jwt.PyJWKClient(jwks_url or f"{self.issuer}/.well-known/jwks.json")

            async def key_provider(token: str) -> object:
                # PyJWKClient caches keys but fetches synchronously, so keep it off the loop.
                return (await asyncio.to_thread(client.get_signing_key_from_jwt, token)).key

        self._key_provider = key_provider

    @classmethod
    def from_settings(cls, settings: Settings) -> TokenVerifier:
        if not settings.auth_issuer or not settings.auth_client_id:
            raise ValueError(
                "AUTH_ISSUER and AUTH_CLIENT_ID are required when AUTH_ENABLED=true. "
                "Set AUTH_ENABLED=false only for local development."
            )
        return cls(
            issuer=settings.auth_issuer,
            client_id=settings.auth_client_id,
            jwks_url=settings.auth_jwks_url,
            allowed_groups=_split(settings.auth_allowed_groups),
            admin_group=settings.auth_admin_group,
        )

    async def verify(self, token: str) -> User:
        try:
            key = await self._key_provider(token)
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self.issuer,
                options={"require": ["exp", "iss", "sub"], "verify_aud": False},
            )
        except Exception as error:  # signature, expiry, issuer, unknown key, JWKS fetch
            raise HTTPException(401, "Invalid or expired token.") from error

        # Cognito access tokens carry client_id; ID tokens carry aud. Accept only our client,
        # and never an ID token as an API credential.
        if claims.get("token_use", "access") != "access" or (
            claims.get("client_id", claims.get("aud")) != self.client_id
        ):
            raise HTTPException(401, "Invalid or expired token.")

        groups = frozenset(claims.get("cognito:groups", claims.get("groups", [])))
        if not groups & self.allowed_groups:
            raise HTTPException(403, "Your account is not allowed to use this service.")
        return User(
            id=str(claims["sub"]), groups=groups, is_admin=self.admin_group in groups
        )


async def get_current_user(request: Request) -> User:
    """FastAPI dependency: the authenticated user, or a dev stand-in when auth is disabled."""
    verifier: TokenVerifier | None = request.app.state.auth
    if verifier is None:
        return ANONYMOUS
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            401, "Authentication required.", headers={"WWW-Authenticate": "Bearer"}
        )
    return await verifier.verify(token.strip())
