# Progress tracker

The live status of the project. `production-plan.md` holds the architecture and the reasoning;
this file holds what is done, what is not, and what each open item waits on. **Read this first
when asked for progress, and update it whenever work is finished.**

Last updated: 2026-11-24 (after commit `8425a7d`, guardrails).

Legend: `[x]` done and verified in development · `[~]` built, not verified for production ·
`[ ]` not done. **Who** is `claude` (can be done in the repo), `you` (needs your AWS account,
money or a decision), or `both`.

## Standing rules

- No Bedrock or other paid AWS call without an explicit request. `BEDROCK_CALLS_ENABLED=true`
  is set only by the user, per command (see `aws.py`).
- Real `.env*` files are never committed. Only the `*.example` files are.
- `uv run` can rewrite `uv.lock`; revert it with `git checkout uv.lock` unless dependencies
  changed on purpose.

## A. Done

- [x] LangGraph workflow: supervisor, finance, academic, clarification, unsupported, final response
- [x] FastAPI API and browser UI, with per-user request ownership
- [x] Fee database, demo seed and read-only MCP `get_fees` tool; fee lists by billing frequency
- [x] Academic RAG: PDF ingest, incremental re-embedding, hybrid vector + keyword search, reranker
- [x] Ollama and Bedrock for supervisor, final response, academic agent, embeddings and rerank
- [x] Profiles (`APP_PROFILE` selects `.env.ollama` or `.env.bedrock`)
- [x] Explicit AWS credentials and the `BEDROCK_CALLS_ENABLED` opt-in
- [x] Auth server side (JWT, groups, admin-only ingest), tested against a local JWKS endpoint
- [x] Rate limit (429) and model concurrency cap (503)
- [x] Postgres checkpointer, elicitation broker, request ownership and rate-limit tables
- [x] Persistent MCP session
- [x] Evaluation runner, 36-case dataset, Ragas baseline on 27 cases
- [x] Tests: 168 unit tests plus 9 Postgres integration test files
- [x] Guardrails (`guardrails.py`): local check always on, Bedrock `ApplyGuardrail` after it
      when `GUARDRAIL_ID` is set and the opt-in is on; applied to queries, clarification
      replies and final answers; 9 unit tests

## B. Not done, with dependencies

### Quick fixes (no dependencies)

| # | Item | Who | Notes |
|---|------|-----|-------|
| B1 | [ ] Fix mypy error `api.py:487` (`screened.message` can be `None`) | claude | My own change. 25 other mypy errors predate it. |
| B2 | [ ] Clear the remaining mypy and ruff errors (about 25 and 44) | claude | Needed before CI can enforce them (D1). |
| B3 | [ ] Add a guardrails section to the README | claude | |
| B4 | [ ] Fix stale statements in `production-plan.md` | claude | Done partly on 2026-11-24; re-check after each change. |
| B5 | [ ] Check `static/app.js` shows the guardrail 422 message properly | claude | Browser UI was not touched when guardrails were added. |

### Evaluation and quality

| # | Item | Who | Depends on |
|---|------|-----|------------|
| C1 | [ ] Eval cases: fee-list by frequency, guardrail blocks and masking | claude | none |
| C2 | [ ] Fresh Ragas run on all cases | you (paid if Bedrock judge) | C1; opt-in |
| C3 | [ ] Set quality thresholds as release gates | both | C2; content-owner review of labels |
| C4 | [ ] Human review of low-scoring Ragas cases (BOOK-007, RAG-004, BOOK-009) | you | C2 |
| C5 | [ ] Guardrail eval: confirm injection and PII cases behave under Bedrock | you (paid) | A guardrail exists (E3); C1 |

### Build and deploy

| # | Item | Who | Depends on |
|---|------|-----|------------|
| D1 | [ ] GitHub Actions: ruff, mypy, unit tests | claude | B1, B2 (or accept non-blocking lint first) |
| D2 | [ ] Dockerfile for the API (and MCP server) | claude | none |
| D3 | [ ] Host the MCP server as its own authenticated service | both | D2; deployment target chosen |
| D4 | [ ] Secrets store for credentials; IAM role instead of keys | you | deployment target chosen |
| D5 | [ ] Database migrations run as a deploy step; backup, restore, rollback | both | deployment target chosen |
| D6 | [ ] Integration tests in CI (needs Postgres service) | claude | D1 |

### Needs your AWS account or a decision

| # | Item | Who | Depends on |
|---|------|-----|------------|
| E1 | [ ] AWS budget alert | you | none (do this before any paid run) |
| E2 | [ ] Real Cognito test: user pool, hosted login, token refresh | you | E1; pool created |
| E3 | [ ] Create the Bedrock guardrail with `scripts/create_bedrock_guardrail.py`, review filters and topics, set `GUARDRAIL_ID` in `.env.bedrock` | you | E1; IAM permission `bedrock:CreateGuardrail` |
| E4 | [ ] Production database ingested with Titan embeddings | you (paid) | E1; school-approved documents (F1) |
| E5 | [ ] IAM policy review for the app (`docs/bedrock-iam-policy.json`; add `bedrock:ApplyGuardrail`) | both | E3 |

### Content, product and security decisions

| # | Item | Who | Depends on |
|---|------|-----|------------|
| F1 | [ ] School-approved fee, syllabus and book data (current data is demo) | you | school content owners |
| F2 | [ ] Citation and source display policy | you | product decision |
| F3 | [ ] Threat model: retrieved-document injection, data retention, log and trace redaction | both | none to start |
| F4 | [ ] Output guardrail for retrieved text before it reaches the model (not covered today) | claude | F3 |
| F5 | [ ] Token refresh and cleanup of old `request_owners` rows | claude | E2 for the refresh part |

### Operations

| # | Item | Who | Depends on |
|---|------|-----|------------|
| G1 | [ ] Load and soak tests; tune rate and concurrency limits; limit MCP-side embedding calls | both | D2, E4, agreed service objectives |
| G2 | [ ] LangSmith tracing confirmed, metrics, alerts, runbooks | both | D3 |
| G3 | [ ] Cancellation, idempotency and repeated-tool-call protection verified | claude | none |
| G4 | [ ] Define service objectives (traffic, availability, latency) | you | none |

## C. Suggested order

1. B1–B5 and C1 (all `claude`, no dependencies).
2. D2, then D1 and D6.
3. E1 (before any paid step), then E3, E2, C5, C2.
4. F1–F3 in parallel with the above, since they depend on people rather than code.
5. E4, G1, G2 once the deployment target (D3–D5) is chosen.

## D. Change log

- 2026-11-24: guardrails added and pushed (`8425a7d`). Progress tracker created.
- Earlier: see `git log`.
