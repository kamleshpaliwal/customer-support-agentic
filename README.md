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
├── db/         # PostgreSQL access, migrations, initialization, and CSV import
├── mcp/        # MCP client and fee lookup server
├── api.py      # FastAPI app (query, resume, elicitation, ingestion routes)
├── graph.py    # LangGraph workflow and routing edges
├── checkpointer.py  # PostgreSQL checkpointer for paused sessions
├── schemas.py  # shared request, decision, and result models
└── settings.py # environment-backed configuration
```

## Fee database and MCP server

PostgreSQL stores structured `fees` records and academic PDF passages with pgvector embeddings. The MCP server exposes `get_fees` for exact fee lookups and `search_academic_documents` for semantic search over the indexed academic sources. Academic sources share one table and are separated by the `document_type` metadata value (`syllabus` or `book_recommendations`).

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

The endpoint accepts an optional JSON `document_type`: `syllabus` (the default) or `book_recommendations`. It selects the matching server-configured path; callers cannot submit arbitrary filesystem paths. Syllabus passages are structure-chunked by standard. Book-reference pages are combined into one chunk per standard-and-subject recommendation record, with page range and document type stored in metadata. The importer creates embeddings through Ollama's local `/api/embed` endpoint and saves or replaces only the selected source's chunks in PostgreSQL. The response reports document type, file name and number of chunks indexed. Repeating the request reindexes that source. The MCP `search_academic_documents` tool accepts the same `document_type` filter and searches matching embeddings by cosine distance. A LangChain `create_agent` Academic Specialist autonomously selects the syllabus or book-recommendations MCP search tool. If the agent sees an unclear request or one outside both sources, it calls neither tool and asks the user to clarify. Retrieved passages, grade, subject and page references are passed to the final responder as evidence.

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

Configure `OLLAMA_EMBEDDING_MODEL`, `ACADEMIC_AGENT_OLLAMA_MODEL`, `ACADEMIC_DOCUMENT_PATH`, `ACADEMIC_BOOK_RECOMMENDATIONS_DOCUMENT_PATH`, `ACADEMIC_INGEST_API_KEY`, `ACADEMIC_SEARCH_TOP_K`, and `ACADEMIC_SEARCH_MIN_SIMILARITY` in `.env`. The configured embedding model must return 768 dimensions to match the vector column. Use the same embedding model for ingestion and queries. If the model changes, re-embed the documents and migrate the vector column before using them.

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

Choose the provider for each role independently with `SUPERVISOR_PROVIDER` and `FINAL_RESPONSE_PROVIDER`, each set to `ollama` or `bedrock`. For Bedrock, configure `AWS_REGION`, `SUPERVISOR_BEDROCK_MODEL_ID`, and/or `FINAL_RESPONSE_BEDROCK_MODEL_ID`, and provide AWS credentials through the standard credential chain or an attached IAM role. Both Bedrock model IDs default to `us.amazon.nova-lite-v1:0`; model availability and pricing depend on the AWS account and region. Amazon lists the model and inference profile IDs in its [Nova Lite documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html).

For example, you can keep supervisor routing local while using Bedrock only for final wording by setting `SUPERVISOR_PROVIDER=ollama` and `FINAL_RESPONSE_PROVIDER=bedrock`.

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

Install or refresh the project dependencies, then start the local API:

```bash
python -m pip install -e .
python -m support_agents.api
```

The local server listens at `http://127.0.0.1:5000`. Open `http://127.0.0.1:5000/` for the browser interface. Its stylesheet and JavaScript live in the project-level `static/` folder, outside the `support_agents` package. The Finance and Academic tabs provide relevant prompts and examples; each question is submitted to `/api/v1/query`, and the supervisor continues to classify and route it. Check the API with `GET /health`, or submit a request:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the tuition fee for CBSE class 1 in 2026-2027?"}'
```

The `user_id` field is optional and defaults to `anonymous`. A completed response contains `request_id`, `status`, `answer`, `intent`, and `limitations`. If the supervisor needs more information, the API returns `status: "needs_input"`, a `question`, and the same `request_id`. Submit the user's reply to the resume endpoint:

```bash
REQUEST_ID='paste-the-request_id-from-the-previous-response'
curl -X POST "http://127.0.0.1:5000/api/v1/query/${REQUEST_ID}/resume" \
  -H 'Content-Type: application/json' \
  -d '{"answer":"2026-2027"}'
```

### Paused sessions (checkpointing)

When a question needs clarification, the LangGraph run pauses and its state is saved so `POST /api/v1/query/<request_id>/resume` can continue it. By default (`CHECKPOINT_BACKEND=postgres`) the state is stored in the same PostgreSQL database as `DATABASE_URL`; the checkpoint tables are created automatically at startup. Paused sessions therefore survive a server restart and can be resumed by any process. Set `CHECKPOINT_BACKEND=memory` only for tests or throwaway single-process development, because those sessions (and pending elicitation questions) are lost on restart and not shared between workers.

### Running the API in deployment

```bash
uvicorn support_agents.api:create_app --factory --host 0.0.0.0 --port 5000
```

Multiple workers (`--workers N`) are safe with the default `CHECKPOINT_BACKEND=postgres`, because both paused graph runs and pending MCP elicitation questions live in PostgreSQL. Run `python -m support_agents.db.init_db` after upgrading to create the new table. The API has no user authentication yet; see `docs/authentication.md` and keep it on a protected network.

### Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

Unit tests use fakes and need no services. `tests/integration/test_postgres_checkpointer.py` needs the Compose Postgres running and skips itself otherwise.

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
