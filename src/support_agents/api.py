"""FastAPI HTTP API for submitting customer support requests to LangGraph."""

from __future__ import annotations

import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal, TypeVar
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from support_agents.agents.academic_specialist import AcademicSpecialist
from support_agents.agents.final_responder import create_final_responder
from support_agents.agents.finance_specialist import FinanceSpecialist
from support_agents.agents.supervisor import create_intent_detector
from support_agents.checkpointer import close_checkpointer, create_checkpointer
from support_agents.db.connection import create_pool
from support_agents.db.ingest_academic_pdf import ingest_configured_pdf
from support_agents.graph import GraphDependencies, build_support_graph
from support_agents.mcp.elicitation import (
    ElicitationBroker,
    InMemoryElicitationBroker,
    PostgresElicitationBroker,
    set_elicitation_broker,
)
from support_agents.settings import PROJECT_ROOT, configure_langsmith_tracing, get_settings

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
TEMPLATE_PATH = PROJECT_ROOT / "src" / "support_agents" / "templates" / "index.html"


class QueryPayload(BaseModel):
    """Public request body accepted by the query endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=20_000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=200)
    request_id: str | None = None

    @field_validator("query", "user_id")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str | None) -> str | None:
        return str(UUID(value)) if value else None


class ClarificationPayload(BaseModel):
    """User's reply to a clarification requested by a paused graph run."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=2_000)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        return value.strip()


class ElicitationAnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=100)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        return value.strip()


class AcademicIngestPayload(BaseModel):
    """Select one of the server-configured academic PDFs for ingestion."""

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["syllabus", "book_recommendations"] = "syllabus"


def _error(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


async def _parse_body(
    request: Request, model: type[ModelT], error_message: str, **extra
) -> ModelT | JSONResponse:
    """Validate a JSON body, returning the model or the API's standard error response."""
    if not request.headers.get("content-type", "").startswith("application/json"):
        return _error(415, "Content-Type must be application/json.", **extra)
    try:
        return model.model_validate(json.loads(await request.body()))
    except ValidationError as error:
        return _error(
            400,
            error_message,
            details=error.errors(include_url=False, include_context=False),
            **extra,
        )
    except (ValueError, TypeError):
        return _error(400, "Invalid JSON request body.", **extra)


def _valid_uuid(value: str) -> str | None:
    try:
        return str(UUID(value))
    except ValueError:
        return None


def _query_response(result: dict, request_id: str) -> JSONResponse:
    """Serialize either a completed graph result or a pending clarification."""
    pending = result.get("__interrupt__", ())
    if pending:
        payload = pending[0].value
        return JSONResponse(
            {
                "request_id": request_id,
                "status": "needs_input",
                "kind": payload.get("kind", "clarification"),
                "intent": payload.get("intent", "clarification"),
                "question": payload.get("question", "Please clarify your request."),
                "options": payload.get("options", []),
            }
        )
    return JSONResponse(
        {
            "request_id": request_id,
            "status": "completed",
            "answer": result["answer"],
            "intent": result["intent"],
            "limitations": result.get("limitations", []),
        }
    )


def create_app(*, graph=None, broker: ElicitationBroker | None = None) -> FastAPI:
    """Create the app. Graph, checkpointer and broker are built on the server's event loop.

    Passing ``graph`` (and optionally ``broker``) skips the real wiring, for tests.
    """

    configure_langsmith_tracing()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        checkpointer = None
        pool = None
        if graph is not None:
            app.state.graph = graph
            app.state.broker = broker or InMemoryElicitationBroker()
        else:
            checkpointer = await create_checkpointer()
            if get_settings().checkpoint_backend == "postgres":
                pool = await create_pool()
                app.state.broker = PostgresElicitationBroker(pool)
            else:
                app.state.broker = InMemoryElicitationBroker()
            app.state.graph = build_support_graph(
                GraphDependencies(
                    intent_detector=create_intent_detector(),
                    finance_specialist=FinanceSpecialist(),
                    academic_specialist=AcademicSpecialist(),
                    final_responder=create_final_responder(),
                ),
                checkpointer=checkpointer,
            )
        set_elicitation_broker(app.state.broker)
        try:
            yield
        finally:
            if checkpointer is not None:
                await close_checkpointer(checkpointer)
            if pool is not None:
                await pool.close()

    app = FastAPI(title="Customer Support Agent System", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

    @app.get("/")
    async def home():
        return FileResponse(TEMPLATE_PATH, media_type="text/html")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/elicitation/{request_id}")
    async def get_pending_elicitation(request_id: str):
        request_id = _valid_uuid(request_id)
        if request_id is None:
            return _error(400, "Invalid request ID.")
        pending = await app.state.broker.get_pending(request_id)
        if pending is None:
            return {"status": "waiting", "request_id": request_id}
        return {
            "status": "pending",
            "request_id": request_id,
            "question": pending.message,
            "field": pending.field_name,
            "options": pending.options,
        }

    @app.post("/api/v1/elicitation/{request_id}/answer")
    async def answer_mcp_elicitation(request_id: str, request: Request):
        request_id = _valid_uuid(request_id)
        if request_id is None:
            return _error(400, "Invalid request ID.")
        payload = await _parse_body(
            request, ElicitationAnswerPayload, "Invalid elicitation answer.", request_id=request_id
        )
        if isinstance(payload, JSONResponse):
            return payload
        try:
            result = await app.state.broker.submit(request_id, payload.answer)
        except LookupError as error:
            return _error(404, str(error), request_id=request_id)
        except ValueError as error:
            return _error(400, str(error), request_id=request_id)
        return {"status": result, "request_id": request_id}

    @app.post("/api/v1/academic/ingest")
    async def ingest_academic_document(request: Request):
        expected_key = get_settings().academic_ingest_api_key
        if not expected_key:
            return _error(503, "Document ingestion is disabled; configure ACADEMIC_INGEST_API_KEY.")

        scheme, separator, supplied_key = request.headers.get("authorization", "").partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(supplied_key.strip(), expected_key)
        ):
            return _error(401, "A valid bearer token is required.")

        raw_body = await request.body()
        if raw_body and not request.headers.get("content-type", "").startswith("application/json"):
            return _error(415, "Content-Type must be application/json when a body is provided.")
        try:
            payload = AcademicIngestPayload.model_validate(json.loads(raw_body) if raw_body else {})
        except ValidationError as error:
            return _error(
                400,
                "Invalid ingestion request.",
                details=error.errors(include_url=False, include_context=False),
            )
        except (ValueError, TypeError):
            return _error(400, "Invalid ingestion request.", details=None)

        try:
            pdf_path, chunk_count = await ingest_configured_pdf(
                document_type=payload.document_type
            )
        except FileNotFoundError:
            logger.exception("Configured academic PDF was not found")
            return _error(404, "The configured academic PDF was not found.")
        except Exception:
            logger.exception("Academic document ingestion failed")
            return _error(502, "Academic document ingestion failed.")

        return {
            "status": "completed",
            "document_type": payload.document_type,
            "source_file": pdf_path.name,
            "chunks_indexed": chunk_count,
        }

    @app.post("/api/v1/query")
    async def submit_query(request: Request):
        payload = await _parse_body(
            request, QueryPayload, "Invalid request body.", request_id=str(uuid4())
        )
        if isinstance(payload, JSONResponse):
            return payload

        browser_request_id = payload.request_id is not None
        request_id = payload.request_id or str(uuid4())

        if not payload.query or not payload.user_id:
            return _error(400, "query and user_id must not be blank.", request_id=request_id)

        try:
            result = await app.state.graph.ainvoke(
                {
                    "request_id": request_id,
                    "user_id": payload.user_id,
                    "query": payload.query,
                    "allow_mcp_elicitation": browser_request_id,
                },
                config={"configurable": {"thread_id": request_id}},
            )
        except Exception:
            logger.exception("Support graph request failed; request_id=%s", request_id)
            return _error(
                502,
                "The support request could not be processed right now.",
                request_id=request_id,
            )
        return _query_response(result, request_id)

    @app.post("/api/v1/query/{request_id}/resume")
    async def resume_query(request_id: str, request: Request):
        request_id = _valid_uuid(request_id)
        if request_id is None:
            return _error(400, "Invalid request ID.")
        payload = await _parse_body(
            request, ClarificationPayload, "Invalid clarification response.", request_id=request_id
        )
        if isinstance(payload, JSONResponse):
            return payload
        if not payload.answer:
            return _error(400, "answer must not be blank.", request_id=request_id)

        try:
            result = await app.state.graph.ainvoke(
                Command(resume=payload.answer),
                config={"configurable": {"thread_id": request_id}},
            )
        except Exception:
            logger.exception("Could not resume support graph; request_id=%s", request_id)
            return _error(
                409,
                "This clarification session could not be resumed. It may have expired; "
                "please submit your question again.",
                request_id=request_id,
            )
        return _query_response(result, request_id)

    return app


if __name__ == "__main__":
    # Local development entry point. For deployment:
    #   uvicorn support_agents.api:create_app --factory
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=5000)
