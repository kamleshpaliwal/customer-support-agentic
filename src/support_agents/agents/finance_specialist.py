"""Finance specialist backed by the read-only fees MCP tool."""

import logging
import re

from support_agents.mcp.client import FeeMCPClientError, FeesMCPClient
from support_agents.schemas import Evidence, FeeLookupRequest, SpecialistResult

logger = logging.getLogger(__name__)


class FinanceSpecialist:
    """Fetch verified fee records through MCP and return evidence for the final responder."""

    def __init__(self, fees_client: FeesMCPClient | None = None) -> None:
        self.fees_client = fees_client or FeesMCPClient()

    async def respond(
        self,
        query: str,
        *,
        user_id: str,
        fee_lookup: FeeLookupRequest | None = None,
        request_id: str | None = None,
    ) -> SpecialistResult:
        del user_id, request_id  # The structured lookup is produced by the supervisor.
        if fee_lookup is None:
            return SpecialistResult(
                specialist="finance",
                answer="Please provide the education board, standard, and academic year.",
                limitations=["A complete fee lookup request was not provided."],
                needs_follow_up=True,
            )

        try:
            result = await self.fees_client.get_fees(fee_lookup)
            # If the model shortened a board label, trust an exact candidate explicitly
            # written in the user's query. This is not a fuzzy substitution: the user
            # already named this full board label, and MCP found it for this class/year.
            if not result.found and result.board_candidates:
                explicitly_named = [
                    candidate
                    for candidate in result.board_candidates
                    if re.search(
                        rf"(?<![\w]){re.escape(candidate)}(?![\w])",
                        query,
                        flags=re.IGNORECASE,
                    )
                ]
                if len(explicitly_named) == 1:
                    fee_lookup = fee_lookup.model_copy(
                        update={"board": explicitly_named[0]}
                    )
                    result = await self.fees_client.get_fees(fee_lookup)
        except FeeMCPClientError as error:
            return SpecialistResult(
                specialist="finance",
                answer="I could not retrieve the fee records right now.",
                limitations=[str(error)],
            )

        if not result.found:
            logger.warning(
                "Fee MCP found no row for board=%r, standard=%r, academic_year=%s",
                fee_lookup.board,
                fee_lookup.standard,
                fee_lookup.academic_year,
            )
            if result.board_candidates:
                candidates = ", ".join(result.board_candidates)
                question = (
                    f"I couldn't find an exact record for {fee_lookup.board}. "
                    f"I found fee records for {candidates}. Which board should I use? "
                    "Reply with one of those board names, or say no to cancel the lookup."
                )
                return SpecialistResult(
                    specialist="finance",
                    answer=question,
                    needs_follow_up=True,
                    follow_up_options=result.board_candidates,
                )

            return SpecialistResult(
                specialist="finance",
                answer=result.message,
                limitations=[result.message],
            )

        evidence = [
            Evidence(
                source="MCP get_fees",
                detail=(
                    f"{item.fee_type}: {item.amount} {item.currency} "
                    f"({item.billing_frequency})"
                ),
            )
            for item in result.fee_items
        ]
        if result.total_amount is not None and result.currency and result.billing_frequency:
            evidence.append(
                Evidence(
                    source="MCP get_fees",
                    detail=(
                        f"Total: {result.total_amount} {result.currency} "
                        f"({result.billing_frequency})"
                    ),
                )
            )

        limitations = [] if result.total_amount is not None else [result.message]
        return SpecialistResult(
            specialist="finance",
            answer=(
                f"Fee records for {result.board}, {result.standard}, "
                f"academic year {result.academic_year} were found."
            ),
            evidence=evidence,
            limitations=limitations,
        )
