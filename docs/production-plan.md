# Production plan

## Agreed architecture

1. The user submits a query to the application.
2. The LangGraph supervisor detects intent and produces a typed routing decision.
3. The graph routes finance queries to the finance specialist and academic queries to the academic specialist. Ambiguous and unsupported requests take explicit safe paths.
4. A specialist calls only the MCP tools it is authorized to use. MCP tool inputs and outputs are validated at the application boundary; database authorization remains enforced by the tool/data layer.
5. Each specialist returns a structured result, including evidence and limitations.
6. A final response node converts that result into natural language without introducing unsupported facts.

LangGraph owns the end-to-end control flow. MCP is the tool interface, not the graph orchestrator.

## Implemented first slice

- One PostgreSQL `fees` table with board, standard, academic year, fee type, amount, currency, and billing frequency.
- A schema initializer in `support_agents.db.init_db`.
- A read-only MCP `get_fees` tool with parameterized lookup, exact decimal arithmetic, and explicit not-found / mixed-currency / mixed-frequency responses.
- A Python MCP client used by the LangGraph finance specialist node. The supervisor's typed decision carries the board, standard, and academic year into that node.
- A CSV seed importer that validates verified fee rows, skips existing entries by default, and requires an explicit flag to update them.
- Local PostgreSQL Compose configuration and environment-based connection settings. The MCP server supports stdio for local use and Streamable HTTP for networked use.

The fee data itself must be populated from the school's verified records. The MCP tool currently reads only; it does not provide a fee write operation.

## Delivery phases

### 1. Requirements and threat model

- Identify user types, authentication, authorization, data sensitivity, and retention requirements.
- Define expected traffic, latency and availability objectives, and supported intent categories.
- Decide how mixed, ambiguous, unsupported, and out-of-scope questions are handled.

### 2. Contracts and runtime adapters

- [x] Define request, intent, specialist-result, and final-response schemas.
- [x] Add local Ollama and Amazon Bedrock adapters for supervisor routing and final response generation.
- [x] Add a model-backed supervisor that extracts typed fee lookup fields before routing.
- Add MCP connection lifecycle management for a long-running deployed graph runtime.
- Keep credentials and service identifiers in server-side configuration/secrets.

### 3. LangGraph workflow

- Implement intent detection and conditional routing.
- Finance specialist uses the read-only fee MCP tool; academic specialist remains to be implemented.
- Implement clarification and safe fallback paths.
- [x] Implement the final response adapter with grounding instructions and a specialist-report fallback.

### 4. Reliability, security, and state

- Add bounded retries, timeouts, cancellation, and service error handling.
- Enforce user-level data access in tools and storage; never rely on prompts for authorization.
- Choose whether conversation state must persist and define user scoping and retention.
- Redact sensitive content from logs and traces.

### 5. Evaluation and operations

- Build a representative evaluation corpus for routing, tool correctness, grounding, permissions, and response quality.
- Add tracing, metrics, alarms, and operational runbooks.
- Add repeatable deployment environments and rollback procedures.
- Load-test against the agreed service objectives before launch.

## Decisions needed before connecting live services

- Model provider and model(s) for intent detection, specialists, and final response.
- Whether the single-school assumption remains valid, or whether campus/school identifiers are needed.
- User authentication and authorization source.
- Production MCP server hosting, authentication, and network access.
- Whether this application will run on AgentCore and whether any existing AgentCore deployment is to be retained.
- Expected load, latency target, availability target, and conversation persistence requirements.
