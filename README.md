# Customer Support Agent System

A Python customer support application organized as a LangGraph workflow. A supervisor detects the request intent, routes it to a finance or academic specialist, and a final response node turns the specialist's structured result into a user-facing answer. Specialist tools are provided through MCP.

## Request flow

```mermaid
flowchart TD
    U[User query] --> S[Supervisor: detect intent]
    S --> R{Route by intent}
    R -->|Finance| F[Finance specialist]
    R -->|Academic| A[Academic specialist]
    R -->|Unclear / unsupported| C[Clarify or safe fallback]
    F -->|MCP tool call when needed| MF[MCP finance tools]
    A -->|MCP tool call when needed| MA[MCP academic tools]
    MF --> F
    MA --> A
    F --> O[Structured specialist result]
    A --> O
    C --> O
    O --> N[Final response agent]
    N --> U2[Response to user]
```

## Source layout

```text
src/support_agents/
├── agents/     # supervisor, specialists, and final response agent
├── retrieval.py  # academic search: vector/keyword candidates, fusion, rerank
├── reranking.py  # cross-encoder reranker
├── db/         # PostgreSQL access, migrations, initialization, and CSV import
├── mcp/        # MCP server, shared client session, elicitation broker
├── api.py      # FastAPI app (query, resume, elicitation, ingestion routes)
├── graph.py    # LangGraph workflow and routing edges
├── auth.py     # JWT verification and group authorization
├── ownership.py  # which user owns which request
├── limits.py   # per-user rate limit and model concurrency limit
├── checkpointer.py  # PostgreSQL checkpointer for paused sessions
├── schemas.py  # shared request, decision, and result models
└── settings.py # environment-backed configuration
```

## Fee database and MCP server

PostgreSQL stores structured `fees` records and academic PDF passages with pgvector embeddings. The MCP server exposes `get_fees` for exact fee lookups, `list_fees_by_frequency` for fee lists, and `search_academic_documents` for semantic search over the indexed academic sources. Academic sources share one table and are separated by the `document_type` metadata value (`syllabus` or `book_recommendations`).

### Listing fees by billing frequency

A question such as "What fees are charged annually?" needs no board, class or year. The supervisor extracts a billing frequency (`annual`, `monthly`, `quarterly`, `term` or `one_time`), and the finance specialist calls the read-only `list_fees_by_frequency` tool:

- If the question names a board, only that board is listed. Class and academic year narrow the list the same way.
- If it names none and several boards have fees with that frequency, the app asks one question with the boards and **All boards** as options (`fee_listing_board_selection`, shown as buttons in the UI). Any other reply cancels the list.
- The answer is rendered from the database rows (grouped by board and class, exact amounts), not reworded by a model. At most 50 records are returned; if there are more, the answer says how many were not shown and asks you to narrow by board, class or year.
- A question that names board, class and academic year together is still a specific `get_fees` lookup, even if it mentions "annual".

### Run locally

Requirements: Python 3.11 or newer and Docker with Compose.

```bash
cd /Users/apple/Documents/customer-support-agent-system
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
docker compose up -d postgres
# Wait until the postgres service shows as healthy:
docker compose ps
python -m support_agents.db.init_db
```

The Compose service uses the pgvector PostgreSQL image. Database migrations create the vector extension, table, and indexes only; they do not read documents or create embeddings. If you already have a local Postgres container using the Compose volume, run `docker compose up -d postgres` after pulling the updated image; the existing database volume is retained.

### Academic RAG document

The syllabus knowledge source is `data/academic/subjects_standards_1_to_10.pdf`, a general CBSE-aligned overview. It is not an official school or year-specific syllabus. The NCERT-first book reference is `data/academic/ncert_book_recommendations_standards_1_to_10_2026-27.pdf`; its titles and editions must be checked against the current NCERT catalogue and the school's list. `ACADEMIC_DOCUMENT_PATH` selects the syllabus PDF, and `ACADEMIC_BOOK_RECOMMENDATIONS_DOCUMENT_PATH` selects the book-reference PDF. Relative paths are resolved from the project root. Keep source PDFs under version control only if you have the rights to distribute them with the application.

Pull the local embedding model once and initialize the database schema:

```bash
ollama pull nomic-embed-text
python -m support_agents.db.init_db
```

Document ingestion is an explicit API operation. Set a private `ACADEMIC_INGEST_API_KEY` in `.env` to enable it, start the API, then call:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/academic/ingest -H 'Authorization: Bearer YOUR_CONFIGURED_INGEST_KEY'
```

The endpoint accepts an optional JSON `document_type`: `syllabus` (the default) or `book_recommendations`. It selects the matching server-configured path; callers cannot submit arbitrary filesystem paths. Syllabus passages are structure-chunked by standard. Book-reference pages are combined into one chunk per standard-and-subject recommendation record, with page range and document type stored in metadata. The importer creates embeddings through Ollama's local `/api/embed` endpoint and saves or replaces only the selected source's chunks in PostgreSQL. The response reports document type, file name and number of chunks indexed. Repeating the request re-syncs that source incrementally (see below). The MCP `search_academic_documents` tool accepts the same `document_type` filter and searches matching embeddings by cosine distance. A LangChain `create_agent` Academic Specialist autonomously selects the syllabus or book-recommendations MCP search tool. If the agent sees an unclear request or one outside both sources, it calls neither tool and asks the user to clarify. Retrieved passages, grade, subject and page references are passed to the final responder as evidence.

#### Incremental re-indexing

Ingestion only re-embeds chunks whose text changed. Each chunk row stores `content_hash` (SHA-256 of its text) next to `embedding_model`; on every run the new chunks are matched to the stored rows by `(content_hash, embedding_model)`:

- unchanged chunk: the stored vector is kept, and only its `chunk_index` and metadata (for example page numbers) are refreshed;
- edited or new chunk: embedded and inserted; the superseded row is deleted;
- chunk no longer in the document: deleted;
- rows embedded with a different `OLLAMA_EMBEDDING_MODEL` never match, so changing the model re-embeds everything.

The response reports `chunks_indexed` (total), `chunks_embedded`, `chunks_reused` and `chunks_removed`. Send `{"force": true}` (or `python -m support_agents.db.ingest_academic_pdf --force`) to re-embed every chunk, for example if you suspect bad vectors. The whole update is one transaction under a per-source advisory lock, so searches see either the old or the new chunks, and the lock plus one pooled connection are held while the changed chunks are embedded. Chunks are cut along document structure (standard / subject records), so an edit normally changes only the chunk that contains it. Migration `005_add_content_hash.sql` backfills hashes for existing rows, so the first run after upgrading already reuses existing vectors; apply it with `python -m support_agents.db.init_db`.

Ingest the syllabus PDF (existing behavior):

```bash
curl -X POST http://127.0.0.1:5000/api/v1/academic/ingest \\
  -H 'Authorization: Bearer YOUR_CONFIGURED_INGEST_KEY'
```

Ingest the book-recommendations PDF:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/academic/ingest \\
  -H 'Authorization: Bearer YOUR_CONFIGURED_INGEST_KEY' \\
  -H 'Content-Type: application/json' \\
  -d '{"document_type":"book_recommendations"}'
```

Configure `OLLAMA_EMBEDDING_MODEL`, `ACADEMIC_AGENT_OLLAMA_MODEL`, `ACADEMIC_DOCUMENT_PATH`, `ACADEMIC_BOOK_RECOMMENDATIONS_DOCUMENT_PATH`, `ACADEMIC_INGEST_API_KEY`, `ACADEMIC_SEARCH_TOP_K`, and `ACADEMIC_SEARCH_MIN_SIMILARITY` in `.env` (hybrid search and reranking are configured below). The configured embedding model must return 768 dimensions to match the vector column. Use the same embedding model for ingestion and queries. If the model changes, re-embed the documents and migrate the vector column before using them.

### Embeddings: Ollama or Amazon Bedrock

Query and document embeddings come from `create_embedder()` (`embeddings.py`), chosen by `EMBEDDING_PROVIDER`:

- `ollama` (default): the local `OLLAMA_EMBEDDING_MODEL` (`nomic-embed-text`, 768 dimensions).
- `bedrock`: Amazon Titan Text Embeddings V2 (`EMBEDDING_BEDROCK_MODEL_ID`, `EMBEDDING_BEDROCK_DIMENSIONS` = 256, 512 or 1024, default 1024). The IAM policy must allow `bedrock:InvokeModel` on the Titan model (`docs/bedrock-iam-policy.json`).

Vectors from different models, or different sizes, are not comparable. Each row stores its `embedding_model` (for Bedrock it includes the size, for example `amazon.titan-embed-text-v2:0@1024`) and every search filters on the configured model, so rows from another model are ignored, never mixed. The `embedding` column is an untyped `vector` (migration `008`), so one schema serves a 768-dimension local database and a 1024-dimension Bedrock one; the vector index was dropped because it needs a fixed size, which is fine at this corpus size.

**Switching model:** ingestion replaces a source's rows, so a database holds one model per document. Use a separate database per environment, or re-ingest after switching (`POST /api/v1/academic/ingest` with `"force": true`, or the ingest CLI). The MCP server warns at startup if nothing is indexed for the configured model, because academic search would otherwise find nothing. The MCP server process receives `AWS_*` and `EMBEDDING_*` variables, so role credentials on a container platform reach it.

### Retrieval: hybrid search and reranking

Academic search (`src/support_agents/retrieval.py`) runs in stages, each switchable in `.env`:

1. **Candidates.** `vector` mode (default) takes the nearest chunks by embedding similarity (a bi-encoder). `hybrid` mode also runs a PostgreSQL full-text query (`content_tsv`, migration `007`) and merges the two ranked lists with reciprocal rank fusion. Both use the same standard/subject/scope filters.
2. **Rerank (optional).** A cross-encoder reads the query and each candidate together and reorders them. It is slower, so it only sees the `ACADEMIC_CANDIDATE_POOL` best candidates (default 20).
3. **Final cut** to `limit` passages. The `ACADEMIC_SEARCH_MIN_SIMILARITY` threshold uses the cosine score and is applied before reranking, because reranker scores are on a different scale.

```bash
ACADEMIC_RETRIEVAL_MODE=hybrid          # vector | hybrid
ACADEMIC_RERANK_ENABLED=true            # needs: pip install -e '.[rerank]'
ACADEMIC_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

By default (`RERANK_PROVIDER=local`) the reranker model downloads on first use (about 2 GB) and loads when the MCP server starts (about 11 s); each search then costs about 1 s extra for a 17-candidate pool on a laptop. With `RERANK_PROVIDER=bedrock` the Bedrock Rerank API scores the candidates instead (`RERANK_BEDROCK_MODEL_ID`, default `amazon.rerank-v1:0`), so no model or `torch` is needed on the server. Amazon Rerank 1.0 is not offered in `us-east-1`, so `RERANK_BEDROCK_REGION` (default `us-west-2`) is set separately and passages are sent to that region; the IAM policy needs `bedrock:Rerank` and `bedrock:InvokeModel` on the rerank model (`docs/bedrock-iam-policy.json`). If reranking fails, search keeps the first-stage order. New AWS accounts can have very low Rerank API quotas: on the development account, a handful of calls in quick succession were throttled (`ThrottlingException`), so check and raise the Rerank quotas in Service Quotas for the rerank region before relying on `RERANK_PROVIDER=bedrock`, and watch the log for `Reranking failed` lines, which mean search ran without reranking. Postgres full-text ranking has no inverse document frequency (unlike BM25), so common words count as much as rare ones; see the retrieval notes in `evals/README.md`.

Compare strategies on the replayed eval cases (no LLM, seconds to run):

```bash
python evals/retrieval_eval.py --strategy all
python evals/retrieval_eval.py --query "What subjects do students study in Standard 6?"
```

Copy `data/fees_seed.template.csv` to a new CSV file, fill it with the school's verified fee records, and import it:

```bash
cp data/fees_seed.template.csv data/fees_seed.csv
python -m support_agents.db.seed_fees data/fees_seed.csv
```

By default, existing matching fee rows are skipped. Pass `--update-existing` only when the CSV is meant to replace matching records. Do not use example or estimated amounts as real fee data.

For a clearly fictional 20-row dataset to try the lookup flow, use `data/fees_seed.demo.csv`:

```bash
python -m support_agents.db.seed_fees data/fees_seed.demo.csv
```

Every demo board is named `DEMO-CBSE` or `DEMO-ICSE`; the example amounts are synthetic and must never be used as actual school fee information.

Run the MCP server locally over stdio (the default):

```bash
python -m support_agents.mcp.server
```

This process waits for an MCP host to connect over stdin/stdout. For local inspection, start the MCP Inspector in another terminal:

```bash
mcp dev src/support_agents/mcp/server.py
```

To run the server over Streamable HTTP for a networked MCP client, set `MCP_TRANSPORT=streamable-http` in `.env`, then run the same Python command. It listens at `http://127.0.0.1:8000/mcp` by default.

The Compose database credentials are for local development only. Use managed secrets, restricted database roles, TLS, and an appropriately protected network for deployment. The supervisor, academic specialist, and final response model adapters are implemented.

## Supervisor model configuration

`support_agents.agents.supervisor.create_intent_detector()` creates the configured supervisor adapter. The development default is Ollama, which runs the model on your computer and does not send inference requests to a paid model API. Model files use local disk space and inference uses local CPU/GPU resources. The adapter returns JSON constrained by the routing schema and validates all fields before routing.

The shared `support_agents.retry.retry` decorator supports both sync and async functions. It accepts `max_attempts`, `delay_seconds`, `backoff_factor`, and retryable exception types. The Ollama supervisor uses two total attempts for malformed JSON and transient connection errors. For other functions, configure the attempt count at the decorator:

```python
from support_agents.retry import retry

@retry(max_attempts=3, delay_seconds=1, exceptions=(TimeoutError,))
async def call_service():
    ...
```

Install Ollama from [ollama.com/download](https://ollama.com/download), start it, and download the development model:

```bash
ollama pull qwen3:4b-instruct
```

Supervisor and final response settings are independent. By default, `.env.example` selects Ollama for both. Configure `SUPERVISOR_OLLAMA_MODEL` and `FINAL_RESPONSE_OLLAMA_MODEL` separately if you want different local models; both use the shared `OLLAMA_BASE_URL`.

`OLLAMA_REQUEST_TIMEOUT_SECONDS` sets how long the app waits for each local model response (300 seconds by default). The supervisor does not retry a timed-out generation, since immediately repeating a slow inference can add more load. It does retry malformed JSON and transient connection errors.

Choose the provider for each role independently with `SUPERVISOR_PROVIDER`, `ACADEMIC_AGENT_PROVIDER` and `FINAL_RESPONSE_PROVIDER`, each set to `ollama` (the local default) or `bedrock`. For Bedrock, configure `AWS_REGION`, `SUPERVISOR_BEDROCK_MODEL_ID`, `ACADEMIC_AGENT_BEDROCK_MODEL_ID`, and/or `FINAL_RESPONSE_BEDROCK_MODEL_ID`, and provide AWS credentials through the standard credential chain or an attached IAM role. All three Bedrock model IDs default to `us.amazon.nova-lite-v1:0`; the academic agent calls search tools, so its model must support tool use through the Converse API (Nova and Claude models do); model availability and pricing depend on the AWS account and region. Amazon lists the model and inference profile IDs in its [Nova Lite documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html).

### Switching between Ollama and Bedrock (profiles)

`.env` is the shared base. Keep the settings that differ between setups in small profile files and pick one at start-up with `APP_PROFILE`:

```bash
cp .env.ollama.example .env.ollama      # local models, local database
cp .env.bedrock.example .env.bedrock    # Bedrock models, Titan-embedded database

APP_PROFILE=ollama  python -m support_agents.api
BEDROCK_CALLS_ENABLED=true APP_PROFILE=bedrock python -m support_agents.api
```

**Bedrock calls need an explicit opt-in.** Every AWS client in the project is created in `aws.py`, and it refuses to create one unless the process was started with `BEDROCK_CALLS_ENABLED=true` (or `1`/`yes`) in the real environment. The variable is read from the process environment only, never from `.env` files, so the opt-in is per command and an edited file cannot switch it on. Without it, the Bedrock profile (and any `*_PROVIDER=bedrock`) fails fast with a clear error, and nothing can send a paid request by accident. The switch is forwarded to the MCP server process, which makes the embedding calls. Tests build clients around fake keys with the switch on, but `tests/conftest.py` makes any real AWS request fail.

`.env.<profile>` is loaded after `.env`, so its values win; a real environment variable still beats both, and with no `APP_PROFILE` only `.env` is used. The same variable works for the ingest and eval commands, and the MCP server process receives it so it reads the same files. A profile holds the providers, `DATABASE_URL` and that setup's own connection and model settings (the Ollama URL and models in `.env.ollama`; the AWS region, keys, Bedrock model IDs and Titan settings in `.env.bedrock`) (a database holds one embedding model per document, so Bedrock needs its own ingested database). All Bedrock and AWS settings (region, model IDs, Titan and rerank settings, access keys) belong in `.env.bedrock`, and the shared `.env` should hold none of them. Don't leave empty `AWS_ACCESS_KEY_ID=` lines in a file that is loaded after the one holding the real key, because an empty value overrides it. Profile files are gitignored; only the `*.example` templates are committed.

### AWS credentials

All Bedrock clients are created by `support_agents/aws.py`, which makes the credential choice explicit:

- **Local development:** set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in **`.env.bedrock`** (not the shared `.env`, so Ollama-only runs never load them) to the keys of a dedicated IAM user that may only call `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the models you use (see `docs/bedrock-iam-policy.json`). Those keys are used as-is, with no fallback.
- **Deployment:** leave both empty and attach an IAM role to the compute service (ECS task role, App Runner instance role, EC2 instance profile). The role's temporary credentials are found automatically, so no key or role name goes in `.env`.
- **No accidental fallback:** with no keys, the app refuses to use the default AWS CLI credentials (`~/.aws/credentials`) and raises an error that says how to fix it. Setting `AWS_PROFILE` in the shell, or `AWS_ALLOW_CLI_PROFILE=true`, allows a CLI profile on purpose.
- Setting only one of the two keys is an error, and the secret is never printed in logs or settings output.

The academic agent's Bedrock model is a LangChain `ChatBedrockConverse` (`langchain-aws`) and shares the model concurrency limit with the other roles. Query and document embeddings and the optional reranker are still local; see `docs/production-plan.md`. For example, you can keep supervisor routing local while using Bedrock only for final wording by setting `SUPERVISOR_PROVIDER=ollama` and `FINAL_RESPONSE_PROVIDER=bedrock`.

Inject the configured detector when building the graph:

```python
from support_agents.graph import GraphDependencies, build_support_graph
from support_agents.agents.supervisor import create_intent_detector
from support_agents.agents.final_responder import create_final_responder

dependencies = GraphDependencies(
    intent_detector=create_intent_detector(),
    finance_specialist=finance_specialist,
    academic_specialist=academic_specialist,
    final_responder=create_final_responder(),
)
graph = build_support_graph(dependencies)
```

For finance requests, the detector extracts board, standard, and academic year only when the user stated them. Missing details or an invalid year route to clarification instead of guessing. The final response adapter formats the specialist report as concise user-facing language and falls back to the report if the model is unavailable. Academic requests retrieve relevant passages from the indexed PDF through MCP.

## Finance specialist MCP connection

The finance graph node accepts the validated `fee_lookup` produced by the supervisor and passes it to `FinanceSpecialist`. The academic graph node passes the question to `AcademicSpecialist`, which calls `search_academic_documents` over the same stdio MCP server and validates the structured passages before passing them to the final responder.

## HTTP API (FastAPI)

The FastAPI app accepts a user query as JSON and passes it into the compiled LangGraph workflow. The graph runs intent detection first, routes finance requests through the finance specialist and MCP fee tool, and sends specialist results through the final responder.

Install or refresh the project dependencies, then start the local API. Authentication is on by default and the server refuses to start without an identity provider configured, so for local development put `AUTH_ENABLED=false` in `.env` (it is already set in `.env.example`) and see "Authentication" below for the real setup:

```bash
python -m pip install -e .
python -m support_agents.db.init_db   # creates/updates all tables; safe to repeat
python -m support_agents.api
```

The local server listens at `http://127.0.0.1:5000`. Open `http://127.0.0.1:5000/` for the browser interface. Its stylesheet and JavaScript live in the project-level `static/` folder, outside the `support_agents` package. The Finance and Academic tabs provide relevant prompts and examples; each question is submitted to `/api/v1/query`, and the supervisor continues to classify and route it. Check the API with `GET /health`, or submit a request:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the tuition fee for CBSE class 1 in 2026-2027?"}'
```

With `AUTH_ENABLED=false` the `user_id` field is optional and defaults to `anonymous`; with authentication on it is ignored and the user comes from the verified token (see below). A completed response contains `request_id`, `status`, `answer`, `intent`, and `limitations`. If the supervisor needs more information, the API returns `status: "needs_input"`, a `question`, and the same `request_id`. Submit the user's reply to the resume endpoint:

```bash
REQUEST_ID='paste-the-request_id-from-the-previous-response'
curl -X POST "http://127.0.0.1:5000/api/v1/query/${REQUEST_ID}/resume" \
  -H 'Content-Type: application/json' \
  -d '{"answer":"2026-2027"}'
```

### Authentication

Sign-up and login are handled by the identity provider (Amazon Cognito by default); the API only verifies the access token it issues. With `AUTH_ENABLED=true` every `/api/v1/*` route except `/api/v1/auth/config` needs `Authorization: Bearer <access token>`:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/query \
  -H "Authorization: Bearer $ACCESS_TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"What is the tuition fee for DEMO-CBSE Class 5 in 2026-2027?"}'
```

- The token's signature, expiry, issuer and app client are checked (`AUTH_ISSUER`, `AUTH_CLIENT_ID`, optional `AUTH_JWKS_URL`); a missing or invalid token gets `401`.
- Only users in `AUTH_ALLOWED_GROUPS` (default `staff,admin`, read from the `cognito:groups` claim) are allowed, others get `403`. Add `parent` to open it to families. Ingestion needs the `AUTH_ADMIN_GROUP` (default `admin`).
- The user id comes from the token's `sub` claim, never from the request body, and each `request_id` is bound to the user who started it: another user resuming or answering it gets `404`.
- The browser UI signs in through Cognito's hosted login page (`static/auth.js`, authorization code flow with PKCE; set `AUTH_COGNITO_DOMAIN`). This browser flow has not been tested against a real Cognito pool yet.
- `/`, `/health` and `/static/*` stay public. Cognito pool setup and the known gaps are in `docs/authentication.md`.

### Paused sessions (checkpointing)

When a question needs clarification, the LangGraph run pauses and its state is saved so `POST /api/v1/query/<request_id>/resume` can continue it. By default (`CHECKPOINT_BACKEND=postgres`) the state is stored in the same PostgreSQL database as `DATABASE_URL`; the checkpoint tables are created automatically at startup. Paused sessions therefore survive a server restart and can be resumed by any process. Set `CHECKPOINT_BACKEND=memory` only for tests or throwaway single-process development, because those sessions (and pending elicitation questions) are lost on restart and not shared between workers.

### MCP connection

The API starts one MCP server process (stdio) at startup and sends every fee and academic tool call over that single connection (`support_agents/mcp/session.py`), instead of starting a new process per call. A new process costs about 0.6-0.8 s (interpreter start, imports, DB pool) for a query that takes about 2 ms; on a warm session the fee lookup step takes roughly 10-30 ms. The session connects on first use if the startup pre-warm failed, reconnects and retries once if the server process dies (all tools are read-only), and is closed on shutdown. Elicitation prompts (the book-language question) are routed to the request that triggered them. There is one MCP process per API worker, each with its own database pool. Set `MCP_PERSISTENT_SESSION=false` to go back to one process per call, for example when debugging the server. Running the MCP server as a separate network service (`MCP_TRANSPORT=streamable-http`) is still a deployment decision; its authentication and network exposure are not handled yet.

### Overload protection

Two limits protect the model backend and the bill (settings in `.env.example`):

- **Per-user rate limit** (`RATE_LIMIT_REQUESTS` per `RATE_LIMIT_WINDOW_SECONDS`, default 20 per 60 s; `0` disables). It counts `POST /api/v1/query` and `.../resume` for the authenticated user (requests that fail validation are not counted). Over the limit the API returns `429` with a `Retry-After` header. Counts live in the `rate_limits` table, so they are shared by all workers. It is a fixed window, so a user can briefly get up to twice the limit across a window boundary.
- **Model concurrency limit** (`MODEL_MAX_CONCURRENCY`, default 4; `0` disables). At most that many model calls (supervisor, academic agent, final responder) run at once per process; others wait up to `MODEL_QUEUE_TIMEOUT_SECONDS` and then the request gets `503` with `Retry-After: 5`. The limit is per process, so total capacity is `MODEL_MAX_CONCURRENCY x workers`. Only the model call holds a slot, not a whole request, so a request waiting for the user's answer to a book-language question does not block others. If the final responder cannot get a slot, it falls back to the specialist's own text.

The per-user counts need the `rate_limits` table; run `python -m support_agents.db.init_db` after upgrading (it applies migration `006`). Set `MODEL_MAX_CONCURRENCY` to what Ollama (`OLLAMA_NUM_PARALLEL`) or your Bedrock quota can really serve. Embedding calls made by the MCP server subprocess are not limited yet.

### Running the API in deployment

```bash
uvicorn support_agents.api:create_app --factory --host 0.0.0.0 --port 5000
```

Multiple workers (`--workers N`) are safe with the default `CHECKPOINT_BACKEND=postgres`, because both paused graph runs and pending MCP elicitation questions live in PostgreSQL. Run `python -m support_agents.db.init_db` after upgrading: it applies the migrations that create the `pending_elicitations`, `request_owners` and `rate_limits` tables and add `content_hash` to the document chunks (the checkpoint tables are created automatically at startup). Authentication is on by default and the server will not start without `AUTH_ISSUER` and `AUTH_CLIENT_ID`; see `docs/authentication.md` for the Cognito setup. For local development set `AUTH_ENABLED=false` in `.env` and keep the API off any network.

### Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

`tests/unit` uses fakes and needs no services (JWT tests use a locally generated key). `tests/integration` runs against the Compose PostgreSQL and the real MCP server process (checkpointer, shared elicitation broker, request ownership, rate limit, incremental ingestion, persistent MCP session) and skips itself when Postgres is not reachable. Neither suite needs Ollama or AWS. Agent-quality checks are separate: `python evals/run_eval.py` (see `evals/README.md`).

The Finance board-candidate confirmation uses a LangGraph interrupt. For academic book requests that omit language, the academic MCP tool uses MCP elicitation to ask the user to choose English or Hindi; the browser UI polls the pending elicitation endpoint while the original query request waits. The browser supplies a UUID `request_id` so the MCP client's elicitation callback can match the prompt to the browser. Direct API clients that want interactive elicitation must also provide a UUID and poll `/api/v1/elicitation/<request_id>`; submit a choice to `/api/v1/elicitation/<request_id>/answer`. With `CHECKPOINT_BACKEND=postgres` (the default) the pending question is stored in the `pending_elicitations` table (created by `python -m support_agents.db.init_db`), so the query and its poll/answer requests may be served by different workers; unanswered questions expire after 5 minutes. With `memory` the broker is in process memory and needs a single process. After initializing the schema, configure the PDF path and ingestion key, then call the ingestion endpoint before asking academic questions. `python -m support_agents.api` runs a single local uvicorn process. For deployment run `uvicorn support_agents.api:create_app --factory` (add `--workers N` to use more CPU cores; keep `CHECKPOINT_BACKEND=postgres` when you do) behind a protected network configuration.

### Optional LangSmith tracing

LangSmith is optional and tracing is disabled by default. To inspect LangGraph runs and LangChain agent/model/tool calls during development, set your own `LANGSMITH_API_KEY` in the untracked `.env` file, then set:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=customer-support-dev
```

Restart the server after changing these settings. Traces can contain the submitted question, model inputs and outputs, and retrieved passages; start with dummy or redacted data and review your data-handling requirements before enabling tracing for customer requests. Set `LANGSMITH_TRACING=false` to turn it off. `LANGSMITH_ENDPOINT` is optional and only needed for a non-default regional endpoint.

Example academic request after indexing:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/query \
  -H 'Content-Type: application/json' \\
  -d '{"query":"What subjects are taught in Standard 6?"}'
```
