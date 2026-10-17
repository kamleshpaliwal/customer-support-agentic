"""MCP client adapter for the local fees server."""

import asyncio
import sys
from typing import Literal

from mcp import Client, StdioServerParameters
from mcp.types import ElicitRequestFormParams, ElicitResult

from support_agents.schemas import (
    AcademicSearchResult,
    FeeLookupRequest,
    FeeLookupResult,
)
from support_agents.mcp.elicitation import elicitation_broker
from support_agents.settings import PROJECT_ROOT, get_settings


class FeeMCPClientError(RuntimeError):
    """Raised when the MCP fee tool cannot be reached or returns an invalid result."""


class FeesMCPClient:
    """Call the fee MCP server over stdio using the current Python environment."""

    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def get_fees(self, request: FeeLookupRequest) -> FeeLookupResult:
        settings = get_settings()
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "support_agents.mcp.server"],
            cwd=str(PROJECT_ROOT),
            env={"DATABASE_URL": settings.database_url, "MCP_TRANSPORT": "stdio"},
        )

        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with Client(server) as client:
                    result = await client.call_tool(
                        "get_fees",
                        {
                            "board": request.board,
                            "standard": request.standard,
                            "academic_year": request.academic_year,
                        },
                    )
            if result.is_error:
                raise FeeMCPClientError("The fee MCP tool reported an error.")
            if result.structured_content is None:
                raise FeeMCPClientError("The fee MCP tool returned no structured result.")
            return FeeLookupResult.model_validate(result.structured_content)
        except FeeMCPClientError:
            raise
        except TimeoutError as error:
            raise FeeMCPClientError("The fee MCP request timed out.") from error
        except Exception as error:
            raise FeeMCPClientError("Could not connect to the fee MCP server.") from error


class AcademicMCPClientError(RuntimeError):
    """Raised when academic document search fails over MCP."""


class AcademicMCPClient:
    """Call the read-only academic vector search tool on the local MCP server."""

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        settings = get_settings()
        self.timeout_seconds = timeout_seconds or settings.ollama_request_timeout_seconds + 10

    async def search(
        self,
        query: str,
        *,
        document_type: Literal["syllabus", "book_recommendations"] = "syllabus",
        request_id: str | None = None,
    ) -> AcademicSearchResult:
        settings = get_settings()
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "support_agents.mcp.server"],
            cwd=str(PROJECT_ROOT),
            env={"DATABASE_URL": settings.database_url, "MCP_TRANSPORT": "stdio"},
        )

        async def handle_elicitation(context, params) -> ElicitResult:
            del context
            if request_id is None or not isinstance(params, ElicitRequestFormParams):
                return ElicitResult(action="decline")

            properties = params.requested_schema.get("properties", {})
            field_name, field_schema = next(iter(properties.items()), (None, {}))
            options = field_schema.get("enum", []) if isinstance(field_schema, dict) else []
            if not field_name or not options:
                return ElicitResult(action="decline")

            return await elicitation_broker.ask(
                request_id,
                message=params.message,
                field_name=field_name,
                options=[str(option) for option in options],
            )

        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with Client(
                    server,
                    elicitation_callback=handle_elicitation,
                ) as client:
                    result = await client.call_tool(
                        "search_academic_documents",
                        {
                            "query": query,
                            "limit": settings.academic_search_top_k,
                            "document_type": document_type,
                        },
                    )
            if result.is_error:
                raise AcademicMCPClientError("The academic MCP tool reported an error.")
            if result.structured_content is None:
                raise AcademicMCPClientError(
                    "The academic MCP tool returned no structured result."
                )
            return AcademicSearchResult.model_validate(result.structured_content)
        except AcademicMCPClientError:
            raise
        except TimeoutError as error:
            raise AcademicMCPClientError("The academic document search timed out.") from error
        except Exception as error:
            raise AcademicMCPClientError(
                "Could not connect to the academic MCP search tool."
            ) from error
