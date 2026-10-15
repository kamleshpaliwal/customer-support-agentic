"""Academic specialist grounded in passages retrieved from the approved PDF guide."""

import logging

from support_agents.mcp.client import AcademicMCPClient, AcademicMCPClientError
from support_agents.schemas import (
    AcademicSearchResult,
    Evidence,
    FeeLookupRequest,
    SpecialistResult,
)

logger = logging.getLogger(__name__)


def _to_specialist_result(search: AcademicSearchResult) -> SpecialistResult:
    if not search.matches:
        return SpecialistResult(
            specialist="academic",
            answer=(
                "I could not find a sufficiently relevant answer in the indexed academic "
                "subject guide. Please check with the school for the exact syllabus."
            ),
            limitations=[search.message],
            needs_follow_up=True,
        )

    evidence = [
        Evidence(
            source=(
                f"{match.source}, page {match.page}"
                if match.page is not None
                else match.source
            ),
            detail=(
                f"{match.standard}: {match.content}"
                if match.standard
                else match.content
            ),
        )
        for match in search.matches
    ]
    return SpecialistResult(
        specialist="academic",
        answer=(
            "Relevant passages were retrieved from the general CBSE-aligned subject guide. "
            "The guide does not define a school-specific or year-specific syllabus."
        ),
        evidence=evidence,
        limitations=[
            "The indexed PDF is a general overview; exact subjects and syllabus details vary by school and year."
        ],
    )


class AcademicSpecialist:
    """Retrieve academic passages through MCP and pass grounded evidence to final response."""

    def __init__(self, academic_client: AcademicMCPClient | None = None) -> None:
        self.academic_client = academic_client or AcademicMCPClient()

    async def respond(
        self,
        query: str,
        *,
        user_id: str,
        fee_lookup: FeeLookupRequest | None = None,
    ) -> SpecialistResult:
        del user_id, fee_lookup
        try:
            search = await self.academic_client.search(query)
        except AcademicMCPClientError as error:
            logger.exception("Academic MCP search failed")
            return SpecialistResult(
                specialist="academic",
                answer="I could not search the academic guide right now. Please try again later.",
                limitations=[str(error)],
            )
        return _to_specialist_result(search)
