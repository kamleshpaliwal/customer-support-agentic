"""Flask HTTP API for submitting customer support requests to LangGraph."""

from __future__ import annotations

import asyncio
import hmac
import logging
from uuid import uuid4

from flask import Flask, jsonify, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from werkzeug.exceptions import BadRequest

from support_agents.agents.academic_specialist import AcademicSpecialist
from support_agents.agents.final_responder import create_final_responder
from support_agents.agents.finance_specialist import FinanceSpecialist
from support_agents.agents.supervisor import create_intent_detector
from support_agents.db.ingest_academic_pdf import ingest_configured_pdf
from support_agents.graph import GraphDependencies, build_support_graph
from support_agents.settings import get_settings

logger = logging.getLogger(__name__)


class QueryPayload(BaseModel):
    """Public request body accepted by the query endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=20_000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=200)

    @field_validator("query", "user_id")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


def create_app(*, graph=None) -> Flask:
    """Create the Flask app, wiring the configured model and MCP adapters by default."""

    app = Flask(__name__)
    support_graph = graph or build_support_graph(
        GraphDependencies(
            intent_detector=create_intent_detector(),
            finance_specialist=FinanceSpecialist(),
            academic_specialist=AcademicSpecialist(),
            final_responder=create_final_responder(),
        )
    )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

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

        try:
            pdf_path, chunk_count = asyncio.run(ingest_configured_pdf())
        except FileNotFoundError:
            logger.exception("Configured academic PDF was not found")
            return jsonify({"error": "The configured academic PDF was not found."}), 404
        except Exception:
            logger.exception("Academic document ingestion failed")
            return jsonify({"error": "Academic document ingestion failed."}), 502

        return jsonify(
            {
                "status": "completed",
                "source_file": pdf_path.name,
                "chunks_indexed": chunk_count,
            }
        )

    @app.post("/api/v1/query")
    def submit_query():
        request_id = str(uuid4())
        if not request.is_json:
            return jsonify(
                {
                    "error": "Content-Type must be application/json.",
                    "request_id": request_id,
                }
            ), 415

        try:
            payload = QueryPayload.model_validate(request.get_json())
        except ValidationError as error:
            return jsonify({
                "error": "Invalid request body.",
                "details": error.errors(include_url=False, include_context=False),
                "request_id": request_id,
            }), 400
        except (BadRequest, TypeError, ValueError):
            return jsonify({"error": "Invalid JSON request body.", "request_id": request_id}), 400

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
                    }
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

        return jsonify({
            "request_id": request_id,
            "answer": result["answer"],
            "intent": result["intent"],
            "limitations": result.get("limitations", []),
        })

    return app


if __name__ == "__main__":
    # Convenient local development entry point. Use a production WSGI server for deployment.
    create_app().run(host="127.0.0.1", port=5000, debug=False)
