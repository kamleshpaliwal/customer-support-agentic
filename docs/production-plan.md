# Customer Support Agent System: Implementation and Production Plan

This document is the source of truth for the current architecture, implemented development capabilities, remaining work, and decisions required before production. “Implemented” means present in the current development code; it does not mean production-ready or fully verified under load.

## Agreed architecture

1. The user submits a query through the UI or API.
2. A LangGraph supervisor detects intent and returns a typed routing decision.
3. LangGraph routes the request to the finance or academic specialist, or to a clarification/unsupported path.
4. Specialists use only their approved MCP tools. Tool inputs and outputs are validated at application boundaries.
5. Specialists return structured reports with evidence and limitations.
6. A final response node writes a natural-language answer from the specialist report and should not add unsupported facts.

LangGraph owns end-to-end orchestration. MCP provides the specialist tool interface.

## Current implementation status

### Implemented in the development application

- Flask API and browser UI with Finance and Academic tabs.
- LangGraph workflow with supervisor intent detection, finance and academic routing, clarification, unsupported handling, and final response nodes.
- Ollama model adapters for local development and Bedrock adapters for supervisor routing and final response generation.
- PostgreSQL fee schema, initializer, demo CSV seed importer, and read-only MCP `get_fees` tool.
- Finance specialist calls the fee MCP tool; typed routing data carries board, standard, and academic year.
- Academic PDFs can be ingested through a configured API endpoint into PostgreSQL/pgvector. Syllabus and book recommendation searches are exposed through MCP.
- Academic specialist uses LangChain `create_agent` to select from the approved academic search tools.
- MCP elicitation can ask for a preferred book language. Finance board confirmation uses a LangGraph interrupt.
- Query and ingestion request schemas, tool-result validation, selected retries/timeouts, and user-facing fallback responses.
- Optional LangSmith configuration and a Ragas-enabled evaluation runner.
- A labeled evaluation dataset currently contains 27 cases. The full dataset has been run with Ragas; see the baseline report below. Quality thresholds and broader dataset coverage remain open.

### Partially implemented or not production-verified

- Academic/finance data is demo or reference content. School-approved fees, curriculum, and book lists need content-owner validation before being presented as authoritative.
- The labeled evaluation set has been expanded to 36 cases and now reports per-intent classification, clarification, tool-success, finance answer-value, and prompt-injection checks. The expanded workflow run is recorded; the expanded Ragas run still needs completion. Thresholds remain provisional and Ragas remains an LLM-judge estimate.
- The current graph checkpointer and MCP elicitation broker are in process memory. They require one running process and are not durable or shared between workers.
- The ingestion route requires a bearer key when enabled, but user query, resume, and elicitation endpoints do not yet implement user authentication/authorization. `user_id` must not be treated as trusted identity while client-supplied.
- LangSmith tracing is configurable, but active tracing, redaction, dashboards, alarms, and operational runbooks have not been confirmed.
- Retries, timeouts, and fallbacks exist in selected paths; system-wide cancellation, idempotency, repeated-tool-call protection, and failure recovery need verification.
- Source evidence is returned internally, but the user-facing source display and citation policy need a product decision and validation.
- No dedicated automated unit/integration/API test suite or load/soak test results have been confirmed.

## Earlier delivery phases: reconciliation

### 1. Requirements and threat model — pending

- Identify user types, authentication and authorization rules, data sensitivity, and retention/deletion requirements.
- Define supported intents, out-of-scope handling, and behavior for mixed or ambiguous requests.
- Set expected traffic, availability, and latency objectives.
- Review prompt injection from user inputs and retrieved documents, data leakage, abuse controls, and tool permissions.

### 2. Contracts and runtime adapters — mostly implemented; production connection lifecycle pending

- Request, intent, specialist-result, and final-response schemas are implemented.
- Local Ollama and Bedrock adapters for supervisor and final response are implemented.
- Model-backed supervisor field extraction and typed routing are implemented.
- MCP clients currently use local stdio connections. Add and operate a managed connection lifecycle for a long-running deployed graph runtime.
- Keep production credentials and service identifiers in a managed secret store and configure least-privilege identities.

### 3. LangGraph workflow and specialist agents — development flow implemented; behavior verification pending

- Intent detection, conditional routing, finance specialist, academic specialist, final response, clarification, and unsupported nodes are implemented.
- The academic LangChain agent selects among syllabus and book-search MCP tools.
- Finance lookup and academic language elicitation are implemented. MCP elicitation currently relies on in-process request state.
- Add regression tests for routing, tool selection/arguments, clarification, cancellations, and safe stopping behavior.

### 4. Reliability, security, and state — partial

- Add and verify bounded retries, timeouts, cancellation, error handling, safe retries, and protection against tool loops.
- Decide whether conversation state must persist. If so, replace in-memory graph state and elicitation broker with durable, user-scoped shared storage and define expiry/deletion behavior.
- Add authentication and authorization; enforce access in the API, tools, and data layer rather than relying on prompts.
- Redact sensitive content in logs and traces; verify prompt-injection defenses and request limits.

### 5. Evaluation and operations — partial

- A 27-case Ragas baseline and 36-case workflow evaluation are recorded. Expanded Ragas scoring, automated software tests, tracing/operations, deployment, and load/soak tests remain.
- Add automated software tests separately from model-quality evaluations.
- Configure and verify LangSmith traces, operational metrics, alerts, and runbooks.
- Add repeatable deployment environments, database backup/restore, migrations, and rollback procedures.
- Load-test and soak-test against agreed service objectives before release.

## Next execution steps

Work through these in order. Preserve evaluation findings as a baseline before making behavior changes.

### 1. Establish the full evaluation baseline — complete

- Full 27-case dataset run with Ragas: `outputs/evaluations/support-eval-20261018-175430.json`.
- Run outcome: 27/27 completed, 0 errors; intent, tool selection, expected action, applicable tool arguments, and fee values all scored 100%.
- Latency: p50 5.151 seconds; p95 16.958 seconds on the local development setup.
- Ragas: 13 cases scored, 0 partial, 2 skipped. Mean context precision 0.9038, context recall 0.8462, faithfulness 0.6513, and answer relevancy 0.7422.
- Manually reviewed low-scoring cases: BOOK-007's answer correctly abstains, but retrieved evidence truncation weakens the evidence used for scoring; RAG-004's answer correctly states the guide is not an exact official syllabus, despite a zero answer-relevancy judge score. BOOK-009 also receives zero context precision/recall despite returning the expected English title. These cases need more reliable human-labeled evidence and review before the Ragas scores are used as gates.
- Ragas outputs are estimates from the local LLM judge, not ground truth. Dataset expansion, score interpretation, and thresholds continue in Steps 2 and 6.

### 2. Expand evaluation coverage and define quality targets — coverage and targets complete; Ragas verification pending

- Expanded the labeled set from 27 to 36 cases with multiple missing finance fields, component-specific and total-fee checks, an additional out-of-scope request, syllabus paraphrases and scope limits, a prompt-injection case, and book-title paraphrase/abstention cases.
- The runner now reports a multiclass confusion matrix and per-intent precision, recall/TPR, and one-vs-rest FPR; tool success; clarification accuracy and unnecessary clarification rate; finance amounts in both tool evidence and final answer; and prompt-injection disclosure checks.
- Proposed initial release targets, reviewed against the expanded workflow results but still provisional until broader human review:

  | Measure | Proposed target | Notes |
  |---|---:|---|
  | Intent macro recall | ≥ 0.95 | No finance-to-academic or academic-to-finance misroutes in the reviewed set. |
  | Per-intent one-vs-rest FPR | ≤ 0.05 | Review the confusion matrix and class support; a small class cannot establish a reliable rate. |
  | Specialist tool selection and argument accuracy | ≥ 0.98 | Zero calls to tools outside the specialist allowlist. |
  | MCP tool success rate | ≥ 0.99 | Count only expected calls that return validated results; include no-result responses as successful tool executions. |
  | Clarification accuracy | ≥ 0.95 | Ask for every required missing field and confirm near-match board candidates before lookup. |
  | Unnecessary clarification rate | ≤ 0.05 | Do not ask when required details are already explicit. |
  | Finance answer amount accuracy | 1.00 | Exact expected amount(s), currency, and requested fee component in the final answer. |
  | RAG context precision / context recall | ≥ 0.80 / ≥ 0.85 | Ragas estimates are advisory; manually review low-score examples and evidence labels. |
  | RAG faithfulness / answer relevancy | ≥ 0.90 / ≥ 0.85 | Require human spot checks because the local judge has shown false-low results. |
  | Labeled answer correctness | ≥ 0.95 | Human-reviewed sample; do not substitute an LLM judge for authoritative content review. |
  | Evaluation request errors | 0 | Any deterministic evaluator-case failure blocks release review until explained. |
  | Local development p95 latency | ≤ 20 seconds | Regression guard only on the same machine/model/data; production SLO remains a Step 9 decision. |

- Keep measures separate; do not collapse them into one score. ROC AUC is intentionally not targeted because the supervisor returns categorical decisions without calibrated confidence scores. Revisit these proposed values after expanded Ragas scoring, additional labeled cases, and content-owner review; do not treat them as production gates yet.
- Expanded workflow report: `outputs/evaluations/support-eval-20261019-151819.json` (36/36 completed, no errors). Intent, tool selection, tool arguments, expected actions, MCP tool success, clarification accuracy, and prompt-injection checks met the proposed targets. Finance exact-answer values were 6/7 (85.7%), below target; FIN-003 exposed malformed amount formatting in the final response. Local p95 was 35.133 seconds, above the proposed 20-second regression target.
- The attempted expanded Ragas run was stopped after the local evaluator stalled during faithfulness scoring. The completed 27-case Ragas baseline remains the reference until expanded scoring completes. Its faithfulness (0.6513) and answer relevancy (0.7422) are below the proposed targets, and the expanded scores are not yet available.
- The targets are not production gates. Review class support, answer labels, Ragas judge reliability, and production workload objectives before making them release criteria.

### 3. Add automated software tests

- Unit-test routing, schema/input validation, fee lookup, retrieval filtering, and error handling.
- Integration-test LangGraph execution, MCP calls, database queries, and elicitation/resume behavior. Mock model calls where practical for repeatability and cost control.
- Add API end-to-end coverage for finance, academic, clarification, unsupported, and dependency failure flows.
- Run these tests in development and before releases; keep them distinct from agent-quality evaluations.

### 4. Define agent autonomy and control requirements

- Specify each agent’s responsibility, permitted decisions and tools, and conditions for clarification, refusal, human escalation, or termination.
- Define and validate state passed between supervisor, specialists, tools, and final responder; reject malformed or unexpected handoffs.
- Bound tool calls, retries, duration, and repeated-call loops. Ensure retried operations are safe and prevent unintended side effects.
- Define behavior for uncertain routing, ambiguity, conflicting evidence, empty retrieval, and low-confidence answers.
- Require grounded answers to distinguish retrieved/database evidence from general guidance and expose useful source information.
- Treat user input and retrieved documents as untrusted. Test that prompt injection cannot alter instructions, permissions, or tool access.
- Decide which situations require a human, what context is handed over, and what status the user sees.
- Add structured traces for routing, tool choice/arguments, tool outcomes, evidence, and final answers; redact sensitive data.

### 5. Harden agents and MCP tools

- Verify specialist tool allowlists and validate all tool arguments and results at application boundaries.
- Exercise empty results, malformed output, timeouts, database/model errors, cancellation, and recovery paths.
- Prevent unsupported facts, unauthorized access, duplicate side effects, and uncontrolled tool loops.
- Investigate the FIN-003 evaluation failure: the fee tool returns valid values, but the final response rendered `52,000.0-0 INR`; ensure monetary formatting preserves exact amounts.

### 6. Improve and verify RAG quality

- Review document coverage, source metadata, chunk boundaries, embedding consistency, and retrieval results against labeled questions.
- Ensure user-facing answers provide useful sources and clearly state when evidence is incomplete or the document is not authoritative for a school/year.
- Add regression cases for semantic near-matches and questions outside indexed coverage.

### 7. Finalize conversation and elicitation behavior

- Decide whether multi-turn history is required; if it is, define storage, user scoping, expiry, retention, and deletion.
- Verify elicitation resume, matching to the correct request, invalid responses, cancellation, expiry, and process restart behavior.
- Ensure a new query or Finance/Academic tab switch does not reuse unrelated query/session state.

### 8. Complete security and data-protection controls

- Choose and implement authentication/authorization for query, resume, elicitation, and ingestion endpoints.
- Validate ingestion files and content; restrict database and MCP operations with least privilege.
- Move production secrets to managed secret storage. Redact sensitive information from application logs and LangSmith traces.
- Add request size/rate limits and defenses against prompt injection through user content and retrieved documents.

### 9. Prepare production operations and release

- Select deployment environment and MCP hosting/transport; deploy behind a production WSGI server with health checks, structured logs, timeouts, and graceful shutdown.
- Configure database backups/restore, schema migration, environment-specific secrets, monitoring/alerts, and rollback procedures.
- Agree on traffic, availability, latency, and resource objectives; load-test concurrent traffic and soak-test sustained traffic, including database pool behavior.
- Require both automated software tests and agent-quality evaluation thresholds as separate release checks; stage rollout and monitor after release.

## Decisions pending

These are product and deployment decisions, tracked separately from implementation steps. Each decision is marked pending until it is agreed; the related plan step shows where it blocks or shapes the work.

| Decision pending | Related step(s) | Status |
|---|---|---|
| Production model/provider for the supervisor, academic specialist, and final responder; cost and latency constraints | 2, 9 | Pending |
| Single-school or multi-school scope; whether school/campus identifiers are needed in data and tool access | 2, 5, 8 | Pending |
| Identity provider and authorization rules for users, staff, and ingestion administrators | 8 | Pending |
| MCP hosting, network exposure, authentication, and long-running client lifecycle | 2, 5, 9 | Pending |
| Retain/deploy on Amazon Bedrock AgentCore or use another hosting approach | 9 | Pending |
| Traffic forecast and service targets for availability, latency, resource use, and recovery | 9 | Pending |
| Whether to persist conversations; if so, user scoping, retention, expiry, and deletion rules | 7, 8 | Pending |
| Who approves fee and academic source content and how often it is reviewed/refreshed | 1, 6 | Pending |
