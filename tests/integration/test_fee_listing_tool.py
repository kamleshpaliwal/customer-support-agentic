"""list_fees_by_frequency against PostgreSQL through the real MCP server process."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import support_agents.mcp.client as client_module
from support_agents.db.connection import create_pool
from support_agents.mcp.client import FeesMCPClient
from support_agents.mcp.session import McpSession
from support_agents.schemas import FeeListingRequest
from tests.integration.test_mcp_session import pytestmark  # noqa: F401


def test_the_tool_filters_orders_and_caps_rows():
    async def scenario():
        pool = await create_pool()
        board = f"TEST-{uuid4().hex[:8]}"
        rows = [
            (board, "Class 10", "2026-2027", "tuition", 100, "annual"),
            (board, "Class 2", "2026-2027", "tuition", 20, "annual"),
            (board, "Class 2", "2026-2027", "bus", 5, "monthly"),
            (board, "Class 2", "2027-2028", "tuition", 22, "annual"),
        ]
        session = McpSession()
        original_session_factory = client_module.get_mcp_session
        try:
            await pool.executemany(
                "INSERT INTO fees (board, standard, academic_year, fee_type, amount, "
                "billing_frequency) VALUES ($1, $2, $3, $4, $5, $6)",
                rows,
            )
            client = FeesMCPClient()
            client_module.get_mcp_session = lambda: session  # use this test's own server process

            annual = await client.list_fees(FeeListingRequest(billing_frequency="annual", board=board))
            # Class 2 sorts before Class 10 (numeric, not alphabetical), and monthly is excluded.
            assert [(i.standard, i.academic_year) for i in annual.items] == [
                ("Class 2", "2026-2027"), ("Class 2", "2027-2028"), ("Class 10", "2026-2027"),
            ]
            assert annual.items[0].amount == "20.00" and annual.total_matches == 3
            assert annual.boards == [board] and not annual.truncated

            monthly = await client.list_fees(FeeListingRequest(billing_frequency="monthly", board=board))
            assert [i.fee_type for i in monthly.items] == ["bus"]

            narrowed = await client.list_fees(FeeListingRequest(
                billing_frequency="annual", board=board.lower(), standard="2", academic_year="2027-2028"))
            assert [i.amount for i in narrowed.items] == ["22.00"]  # case-insensitive board, "2" -> Class 2

            none = await client.list_fees(FeeListingRequest(billing_frequency="quarterly", board=board))
            assert none.items == [] and none.boards == []
        finally:
            await pool.execute("DELETE FROM fees WHERE board = $1", board)
            await pool.close()
            client_module.get_mcp_session = original_session_factory
            await session.close()

    asyncio.run(scenario())
