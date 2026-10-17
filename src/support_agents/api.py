"""Flask HTTP API for submitting customer support requests to LangGraph."""

from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Literal
from uuid import UUID, uuid4

from flask import Flask, jsonify, render_template, request
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from werkzeug.exceptions import BadRequest

from support_agents.agents.academic_specialist import AcademicSpecialist
from support_agents.agents.final_responder import create_final_responder
from support_agents.agents.finance_specialist import FinanceSpecialist
from support_agents.agents.supervisor import create_intent_detector
from support_agents.db.ingest_academic_pdf import ingest_configured_pdf
from support_agents.graph import GraphDependencies, build_support_graph
from support_agents.mcp.elicitation import elicitation_broker
from support_agents.settings import PROJECT_ROOT, configure_langsmith_tracing, get_settings

logger = logging.getLogger(__name__)


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


def create_app(*, graph=None) -> Flask:
    """Create the Flask app, wiring the configured model and MCP adapters by default."""

    configure_langsmith_tracing()
    app = Flask(__name__, static_folder=str(PROJECT_ROOT / "static"))
    support_graph = graph or build_support_graph(
        GraphDependencies(
            intent_detector=create_intent_detector(),
            finance_specialist=FinanceSpecialist(),
            academic_specialist=AcademicSpecialist(),
            final_responder=create_final_responder(),
        )
    )

    def query_response(result: dict, request_id: str):
        """Serialize either a completed graph result or a pending clarification."""
        pending = result.get("__interrupt__", ())
        if pending:
            payload = pending[0].value
            return jsonify(
                {
                    "request_id": request_id,
                    "status": "needs_input",
                    "kind": payload.get("kind", "clarification"),
                    "intent": payload.get("intent", "clarification"),
                    "question": payload.get("question", "Please clarify your request."),
                    "options": payload.get("options", []),
                }
            )

        return jsonify(
            {
                "request_id": request_id,
                "status": "completed",
                "answer": result["answer"],
                "intent": result["intent"],
                "limitations": result.get("limitations", []),
            }
        )

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/v1/elicitation/<request_id>")
    def get_pending_elicitation(request_id: str):
        try:
            request_id = str(UUID(request_id))
        except ValueError:
            return jsonify({"error": "Invalid request ID."}), 400
        pending = elicitation_broker.get_pending(request_id)
        if pending is None:
            return jsonify({"status": "waiting", "request_id": request_id})
        return jsonify(
            {
                "status": "pending",
                "request_id": request_id,
                "question": pending.message,
                "field": pending.field_name,
                "options": pending.options,
            }
        )

    @app.post("/api/v1/elicitation/<request_id>/answer")
    def answer_mcp_elicitation(request_id: str):
        try:
            request_id = str(UUID(request_id))
        except ValueError:
            return jsonify({"error": "Invalid request ID."}), 400
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json."}), 415
        try:
            payload = ElicitationAnswerPayload.model_validate(request.get_json())
        except ValidationError as error:
            return jsonify(
                {
                    "error": "Invalid elicitation answer.",
                    "details": error.errors(include_url=False, include_context=False),
                    "request_id": request_id,
                }
            ), 400
        except (BadRequest, TypeError, ValueError):
            return jsonify({"error": "Invalid JSON request body.", "request_id": request_id}), 400
        try:
            result = elicitation_broker.submit(request_id, payload.answer)
        except LookupError as error:
            return jsonify({"error": str(error), "request_id": request_id}), 404
        except ValueError as error:
            return jsonify({"error": str(error), "request_id": request_id}), 400
        return jsonify({"status": result, "request_id": request_id})

    @app.post("/api/v1/academic/ingest")
    def ingest_academic_document():
        settings = get_settings()
        expected_key = settings.academic_ingest_api_key
        if not expected_key:
            return jsonify(
                {"error": "Document ingestion is disabled; configure ACADEMIC_INGEST_API_KEY."}
            ), 503

        scheme, separator, supplied_key = request.headers.get("Authorization", "").partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(supplied_key.strip(), expected_key)
        ):
            return jsonify({"error": "A valid bearer token is required."}), 401

        if request.data and not request.is_json:
            return jsonify({"error": "Content-Type must be application/json when a body is provided."}), 415
        try:
            payload_data = request.get_json() if request.is_json else {}
            payload = AcademicIngestPayload.model_validate(payload_data)
        except (ValidationError, BadRequest, TypeError, ValueError) as error:
            details = error.errors(include_url=False, include_context=False) if isinstance(error, ValidationError) else None
            return jsonify({"error": "Invalid ingestion request.", "details": details}), 400

        try:
            pdf_path, chunk_count = asyncio.run(
                ingest_configured_pdf(document_type=payload.document_type)
            )
        except FileNotFoundError:
            logger.exception("Configured academic PDF was not found")
            return jsonify({"error": "The configured academic PDF was not found."}), 404
        except Exception:
            logger.exception("Academic document ingestion failed")
            return jsonify({"error": "Academic document ingestion failed."}), 502

        return jsonify(
            {
                "status": "completed",
                "document_type": payload.document_type,
                "source_file": pdf_path.name,
                "chunks_indexed": chunk_count,
            }
        )

    @app.post("/api/v1/query")
    def submit_query():
        if not request.is_json:
            return jsonify(
                {
                    "error": "Content-Type must be application/json.",
                    "request_id": str(uuid4()),
                }
            ), 415

        try:
            payload = QueryPayload.model_validate(request.get_json())
        except ValidationError as error:
            request_id = str(uuid4())
            return jsonify({
                "error": "Invalid request body.",
                "details": error.errors(include_url=False, include_context=False),
                "request_id": request_id,
            }), 400
        except (BadRequest, TypeError, ValueError):
            request_id = str(uuid4())
            return jsonify({"error": "Invalid JSON request body.", "request_id": request_id}), 400

        browser_request_id = payload.request_id is not None
        request_id = payload.request_id or str(uuid4())

        if not payload.query or not payload.user_id:
            return jsonify(
                {
                    "error": "query and user_id must not be blank.",
                    "request_id": request_id,
                }
            ), 400

        try:
            result = asyncio.run(
                support_graph.ainvoke(
                    {
                        "request_id": request_id,
                        "user_id": payload.user_id,
                        "query": payload.query,
                        "allow_mcp_elicitation": browser_request_id,
                    },
                    config={"configurable": {"thread_id": request_id}},
                )
            )
        except Exception:
            logger.exception("Support graph request failed; request_id=%s", request_id)
            return jsonify(
                {
                    "error": "The support request could not be processed right now.",
                    "request_id": request_id,
                }
            ), 502

        return query_response(result, request_id)

    @app.post("/api/v1/query/<request_id>/resume")
    def resume_query(request_id: str):
        try:
            request_id = str(UUID(request_id))
        except ValueError:
            return jsonify({"error": "Invalid request ID."}), 400

        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json."}), 415
        try:
            payload = ClarificationPayload.model_validate(request.get_json())
        except ValidationError as error:
            return jsonify(
                {
                    "error": "Invalid clarification response.",
                    "details": error.errors(include_url=False, include_context=False),
                    "request_id": request_id,
                }
            ), 400
        except (BadRequest, TypeError, ValueError):
            return jsonify({"error": "Invalid JSON request body.", "request_id": request_id}), 400

        if not payload.answer:
            return jsonify({"error": "answer must not be blank.", "request_id": request_id}), 400

        try:
            result = asyncio.run(
                support_graph.ainvoke(
                    Command(resume=payload.answer),
                    config={"configurable": {"thread_id": request_id}},
                )
            )
        except Exception:
            logger.exception("Could not resume support graph; request_id=%s", request_id)
            return jsonify(
                {
                    "error": "This clarification session could not be resumed. It may have expired; please submit your question again.",
                    "request_id": request_id,
                }
            ), 409

        return query_response(result, request_id)

    return app


if __name__ == "__main__":
    # Convenient local development entry point. Use a production WSGI server for deployment.
    create_app().run(host="127.0.0.1", port=5000, debug=False)
