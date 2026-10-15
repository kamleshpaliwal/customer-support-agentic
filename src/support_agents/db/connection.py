"""PostgreSQL connection setup and schema initialization."""

from pathlib import Path

import asyncpg

from support_agents.settings import get_settings


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


async def create_pool() -> asyncpg.Pool:
    settings = get_settings()
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=1,
        max_size=10,
        command_timeout=10,
        max_inactive_connection_lifetime=300,
    )


async def initialize_schema() -> None:
    """Apply the initial schema migration for local setup and first deployment."""

    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    pool = await create_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                for migration in migrations:
                    await connection.execute(migration.read_text(encoding="utf-8"))
    finally:
        await pool.close()
