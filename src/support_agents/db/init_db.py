"""Initialize the PostgreSQL schema with ``python -m support_agents.db.init_db``."""

import asyncio
import logging

from support_agents.db.connection import initialize_schema


async def main() -> None:
    await initialize_schema()
    logging.info("Database schema initialized.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
