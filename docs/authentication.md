# Authentication and credentials

## User-facing API

Sign-up and login are handled by the identity provider (Amazon Cognito by default). The API
never sees passwords; it verifies the **access token** the provider issues.

```
Browser -> Cognito hosted login (PKCE) -> access token (JWT)
        -> Authorization: Bearer <token> -> API verifies it -> request runs as that user
```

Every `/api/v1/*` route except `GET /api/v1/auth/config` requires a valid token. `/`,
`/health` and `/static/*` stay public so the login page can load.

### What is checked

- RS256 signature against the issuer's public keys (`<issuer>/.well-known/jwks.json`, or
  `AUTH_JWKS_URL`), expiry, and issuer.
- `token_use` must be `access` and `client_id` must equal `AUTH_CLIENT_ID`. ID tokens are
  rejected as API credentials.
- The user must be in one of `AUTH_ALLOWED_GROUPS` (`cognito:groups` claim), default
  `staff,admin`. Others get 403. Add `parent` to open the service to families.
- Ingestion additionally requires the `AUTH_ADMIN_GROUP` (default `admin`).

### Identity and session ownership

- `user_id` comes from the token's `sub` claim. A `user_id` in the request body is ignored
  when authentication is enabled.
- Each `request_id` is bound to the user who started it (`request_owners` table). Resuming,
  polling or answering another user's request returns 404, and a request id cannot be
  claimed by a second user.

### Fail closed

`AUTH_ENABLED` defaults to true. With it on, the server refuses to start unless
`AUTH_ISSUER` and `AUTH_CLIENT_ID` are set. For local development without a provider set
`AUTH_ENABLED=false`; every request then runs as `anonymous`, ingestion falls back to the
shared `ACADEMIC_INGEST_API_KEY`, and **the API must not be exposed to a network**.

## Cognito setup (pilot: staff only)

1. Create a **new** user pool (Lite or Essentials tier has the 10,000 monthly-user free
   allowance; Plus has none). Turn **self-service sign-up off** and create staff accounts
   as an admin.
2. Create groups `staff` and `admin`, and add users to them.
3. Create a **public app client** (no client secret) with the authorization code grant,
   scopes `openid email`, and the callback URL of the site root, e.g.
   `http://localhost:5000/` for development and your real domain for deployment.
4. Add a Cognito domain for the hosted login page.
5. Set `AUTH_ISSUER`, `AUTH_CLIENT_ID` and `AUTH_COGNITO_DOMAIN` in the environment.

The browser UI (`static/auth.js`) runs the hosted-login flow with PKCE and keeps the access
token in `sessionStorage`; on a 401 it signs the user in again. This flow has not been run
against a real Cognito pool yet; the server side is covered by tests and by a run against a
local JWKS endpoint.

## Known gaps

- `request_owners` rows are never deleted.
- Rate limiting is per user and per window (see the README, "Overload protection"); there is no spending cap, so budget alarms on the model account are still advisable.
- No logout button or token refresh; users sign in again when the token expires (about an
  hour by default).
- Tokens in `sessionStorage` are readable by any script on the page, so keep the page free
  of third-party scripts.
- The MCP server is still a local subprocess with no network exposure or authentication of
  its own.

## Service credentials

- Runtime settings are loaded from environment variables and the untracked project `.env`
  file. Keep real credentials out of source control.
- PostgreSQL uses `DATABASE_URL`. The credentials shown in `.env.example` and Compose are for
  local development only.
- Bedrock uses AWS's standard credential chain or an attached IAM role; the app configures
  the AWS region and model ID.
- LangSmith tracing is off by default. When enabled, `LANGSMITH_API_KEY` is required. Traces
  may contain user questions, model input/output, and retrieved passages.

## Implementation references

- `src/support_agents/auth.py`: token verification and the `get_current_user` dependency.
- `src/support_agents/ownership.py`: per-user request ownership (memory and PostgreSQL).
- `src/support_agents/api.py`: route protection.
- `static/auth.js`: browser login flow.
- `tests/unit/test_auth.py`: token, group and ownership tests.
