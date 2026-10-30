# Authentication and credentials

## User-facing API

The API does not currently authenticate users. In particular, `POST /api/v1/query`, the clarification resume endpoint, and the MCP elicitation poll and answer endpoints do not require a login, session, or bearer token.

The optional `user_id` in a query is caller-supplied metadata and defaults to `anonymous`; it is not verified as an identity. Likewise, the UUID `request_id` correlates a request with its LangGraph thread and pending elicitation. It is not a credential. Do not expose these endpoints to an untrusted network without adding authentication and authorization.

## Academic ingestion

`POST /api/v1/academic/ingest` is the exception. It requires `ACADEMIC_INGEST_API_KEY` to be configured on the server and supplied as:

```http
Authorization: Bearer <configured-ingest-key>
```

If the key is missing from server configuration, ingestion is disabled. The API compares the supplied value with the configured key using `hmac.compare_digest`. The request can select only a server-configured document type; it cannot supply an arbitrary filesystem path.

## Service credentials

- Runtime settings are loaded from environment variables and the untracked project `.env` file. Keep real credentials out of source control.
- PostgreSQL uses `DATABASE_URL`. The credentials shown in `.env.example` and Compose are for local development only.
- Bedrock uses AWS's standard credential chain or an attached IAM role; the app configures the AWS region and model ID.
- LangSmith tracing is off by default. When enabled, `LANGSMITH_API_KEY` is required. Traces may contain user questions, model input/output, and retrieved passages.

The production plan lists user authentication/authorization and production MCP authentication/network access as open decisions. Before production exposure, define identity and user-level authorization, protect service credentials, and use a production ASGI server (uvicorn) with an appropriately protected network.

## Implementation references

- `src/support_agents/api.py`: query routes, request validation, and the ingestion bearer-key check.
- `src/support_agents/settings.py`: environment-backed configuration and LangSmith key handling.
- `src/support_agents/mcp/client.py`: local MCP subprocess setup and database URL forwarding.
- `docs/production-plan.md`: remaining production security decisions.
